"""Fake banking backend used by the tools.

Deterministic in-memory data plus a configurable transient-failure injector so
tests and demos can exercise the ``transient`` error path predictably.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .errors import ErrorCategory, ToolError


@dataclass
class Transaction:
    txn_id: str
    date: str  # ISO date
    description: str
    amount: float  # negative = charge, positive = credit


@dataclass
class Account:
    account_id: str
    owner: str
    balance: float
    currency: str = "USD"
    frozen: bool = False
    transactions: list[Transaction] = field(default_factory=list)


def _seed_accounts() -> dict[str, Account]:
    return {
        "ACC-1001": Account(
            account_id="ACC-1001",
            owner="Jordan Lee",
            balance=2450.75,
            transactions=[
                Transaction("TXN-9001", "2026-08-01", "Utility bill payment", -120.00),
                Transaction("TXN-9002", "2026-08-03", "Streamify subscription", -19.99),
                Transaction("TXN-9003", "2026-08-03", "Streamify subscription", -19.99),
                Transaction("TXN-9004", "2026-08-07", "Salary deposit", 3200.00),
                Transaction("TXN-9005", "2026-08-10", "Grocery store", -86.42),
            ],
        ),
        "ACC-2002": Account(
            account_id="ACC-2002",
            owner="Sam Rivera",
            balance=110.10,
            frozen=True,
            transactions=[
                Transaction("TXN-9101", "2026-08-05", "Cafe purchase", -8.50),
            ],
        ),
    }


class BankBackend:
    """In-memory backend with per-tool transient failure injection.

    ``inject_transient_failures(tool_name, n)`` makes the next *n* calls that
    tool routes through this backend raise a retryable transient error before
    succeeding — simulating flaky downstream services.
    """

    def __init__(self) -> None:
        self.accounts = _seed_accounts()
        self._pending_failures: dict[str, int] = defaultdict(int)
        self.tickets: list[dict] = []
        self.refunds: list[dict] = []
        self._ticket_seq = 5000

    # -- failure injection ---------------------------------------------------

    def inject_transient_failures(self, tool_name: str, count: int) -> None:
        self._pending_failures[tool_name] = count

    def _maybe_fail(self, tool_name: str) -> None:
        if self._pending_failures[tool_name] > 0:
            self._pending_failures[tool_name] -= 1
            raise ToolError(
                ErrorCategory.TRANSIENT,
                "The banking service is temporarily unavailable (upstream timeout). "
                "This error is transient — retry the same tool call.",
                details={"errorCode": "SERVICE_UNAVAILABLE", "retryAfterSeconds": 1},
            )

    # -- data access ----------------------------------------------------------

    def get_account(self, tool_name: str, account_id: str) -> Account:
        self._maybe_fail(tool_name)
        account = self.accounts.get(account_id)
        if account is None:
            raise ToolError(
                ErrorCategory.VALIDATION,
                f"No account exists with id '{account_id}'. Do not retry with the "
                "same id; ask the user to confirm their account id (format: ACC-NNNN).",
                details={"errorCode": "ACCOUNT_NOT_FOUND", "accountId": account_id},
            )
        return account

    def find_transaction(self, account: Account, txn_id: str) -> Transaction:
        for txn in account.transactions:
            if txn.txn_id == txn_id:
                return txn
        raise ToolError(
            ErrorCategory.VALIDATION,
            f"Transaction '{txn_id}' does not exist on account "
            f"'{account.account_id}'. Do not retry; verify the transaction id "
            "via get_transaction_history first.",
            details={"errorCode": "TXN_NOT_FOUND", "transactionId": txn_id},
        )

    def next_ticket_id(self) -> str:
        self._ticket_seq += 1
        return f"TICKET-{self._ticket_seq}"
