"""Tool definitions and implementations.

The four tools are written MCP-style: each definition is a name + detailed
description + JSON Schema, exactly the shape an MCP server advertises via
``tools/list`` (and the same shape the Anthropic Messages API accepts).
``mcp_server.py`` exposes these same tools over a real MCP transport.

Two tools deliberately overlap — ``get_account_balance`` and
``get_transaction_history`` both read account data — so their descriptions
must draw a sharp boundary to prevent the model from picking the wrong one:

- balance  = a single point-in-time number, no transaction detail
- history  = a list of dated transactions, never the current balance
"""

from __future__ import annotations

from typing import Any, Callable

from .backend import BankBackend
from .errors import ErrorCategory, ToolError, success_payload

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_account_balance",
        "description": (
            "Return the CURRENT balance of one account as a single point-in-time "
            "number, with currency and account status. Use this when the user asks "
            "'how much money do I have', 'what's my balance', or you need to check "
            "available funds before a refund.\n\n"
            "Boundaries: this tool returns ONLY the current balance snapshot — it "
            "never returns individual transactions, payment history, or whether a "
            "specific payment went through. For any question about past activity "
            "('did my payment go through', 'what did I spend last week', 'was I "
            "double-charged') use get_transaction_history instead.\n\n"
            "Errors: returns a structured error with errorCategory=validation if the "
            "account id is unknown (do not retry — confirm the id with the user), or "
            "errorCategory=transient if the banking service times out (retry the "
            "same call)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier, format ACC-NNNN, e.g. ACC-1001.",
                }
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "get_transaction_history",
        "description": (
            "Return the LIST of individual transactions on an account — each with "
            "transaction id, date, description, and signed amount (negative = "
            "charge, positive = credit) — optionally filtered to a date range. Use "
            "this to answer any question about PAST ACTIVITY: 'did my payment on "
            "Aug 1 go through', 'was I charged twice', 'show my recent purchases', "
            "or to look up a transaction id before initiating a refund.\n\n"
            "Boundaries: this tool never returns the current account balance — the "
            "amounts are per-transaction, not a running total. For 'how much money "
            "is in my account right now' use get_account_balance instead. This tool "
            "is read-only; it cannot reverse or refund a charge — use "
            "initiate_refund for that.\n\n"
            "Errors: errorCategory=validation for an unknown account id or a "
            "malformed date (do not retry with the same input); "
            "errorCategory=transient for upstream timeouts (retry the same call)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier, format ACC-NNNN.",
                },
                "start_date": {
                    "type": "string",
                    "description": "Optional inclusive ISO start date (YYYY-MM-DD).",
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional inclusive ISO end date (YYYY-MM-DD).",
                },
            },
            "required": ["account_id"],
        },
    },
    {
        "name": "initiate_refund",
        "description": (
            "Reverse a specific past charge by issuing a refund to the account. "
            "Requires the exact transaction id of the original charge (look it up "
            "with get_transaction_history first if the user only describes the "
            "charge). The refund amount may be at most the absolute value of the "
            "original charge.\n\n"
            "Business rules enforced by this tool and the surrounding platform:\n"
            "- amount must be a positive number no greater than the original charge\n"
            "- the account must not be frozen\n"
            "- refunds above the auto-approval threshold are BLOCKED by policy and "
            "return errorCategory=permission with instructions to escalate via "
            "create_support_ticket. Do not retry a blocked refund and do not split "
            "it into smaller refunds to evade the threshold — escalate instead.\n\n"
            "Errors: errorCategory=validation for bad amounts or unknown "
            "transaction ids (fix the input, don't retry blindly); "
            "errorCategory=permission for policy blocks (escalate, never retry); "
            "errorCategory=transient for service timeouts (retry the same call)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier, format ACC-NNNN.",
                },
                "transaction_id": {
                    "type": "string",
                    "description": "Id of the original charge to refund, e.g. TXN-9002.",
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "Refund amount in the account currency. Positive, and at "
                        "most the absolute value of the original charge."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Short reason for the refund, e.g. 'duplicate charge'.",
                },
            },
            "required": ["account_id", "transaction_id", "amount", "reason"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Create a ticket for the human support team. This is the ESCALATION "
            "path: use it when an action is blocked by policy (e.g. a refund above "
            "the auto-approval threshold returned errorCategory=permission), when "
            "an account is frozen, or when the user asks for something no other "
            "tool can do. It does not move money or change account state itself — "
            "it queues the request for a human.\n\n"
            "Include everything a human agent needs to act without re-asking the "
            "user: the account id, relevant transaction ids and amounts, and what "
            "the user wants. Set priority='high' only for blocked money movements "
            "or frozen-account issues.\n\n"
            "Errors: errorCategory=transient if the ticketing service is briefly "
            "unavailable (retry the same call)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account identifier the ticket concerns.",
                },
                "summary": {
                    "type": "string",
                    "description": "One-line summary of the request.",
                },
                "details": {
                    "type": "string",
                    "description": (
                        "Full context for the human agent: transaction ids, amounts, "
                        "what was attempted, and why it was escalated."
                    ),
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "description": "Ticket priority. Default normal.",
                },
            },
            "required": ["account_id", "summary", "details"],
        },
    },
]


class ToolExecutor:
    """Dispatches validated tool calls against the backend.

    Returns a JSON string on success and raises ToolError on failure; the
    agentic loop converts either into a tool_result block.
    """

    def __init__(self, backend: BankBackend, refund_auto_approval_limit: float = 500.0):
        self.backend = backend
        self.refund_auto_approval_limit = refund_auto_approval_limit
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "get_account_balance": self._get_account_balance,
            "get_transaction_history": self._get_transaction_history,
            "initiate_refund": self._initiate_refund,
            "create_support_ticket": self._create_support_ticket,
        }
        # observable call log for tests / demos
        self.call_log: list[tuple[str, dict[str, Any]]] = []

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        self.call_log.append((tool_name, dict(tool_input)))
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ToolError(
                ErrorCategory.VALIDATION,
                f"Unknown tool '{tool_name}'. Do not retry.",
                details={"errorCode": "UNKNOWN_TOOL"},
            )
        return handler(tool_input)

    # -- handlers --------------------------------------------------------------

    def _get_account_balance(self, args: dict[str, Any]) -> str:
        account = self.backend.get_account("get_account_balance", args["account_id"])
        return success_payload(
            {
                "accountId": account.account_id,
                "balance": round(account.balance, 2),
                "currency": account.currency,
                "status": "frozen" if account.frozen else "active",
            }
        )

    def _get_transaction_history(self, args: dict[str, Any]) -> str:
        account = self.backend.get_account("get_transaction_history", args["account_id"])
        start = args.get("start_date")
        end = args.get("end_date")
        for label, value in (("start_date", start), ("end_date", end)):
            if value is not None and (len(value) != 10 or value[4] != "-" or value[7] != "-"):
                raise ToolError(
                    ErrorCategory.VALIDATION,
                    f"{label} '{value}' is not a valid ISO date (YYYY-MM-DD). "
                    "Fix the date format; do not retry with the same value.",
                    details={"errorCode": "BAD_DATE", "field": label},
                )
        txns = [
            t
            for t in account.transactions
            if (start is None or t.date >= start) and (end is None or t.date <= end)
        ]
        return success_payload(
            {
                "accountId": account.account_id,
                "transactions": [vars(t) for t in txns],
                "count": len(txns),
            }
        )

    def _initiate_refund(self, args: dict[str, Any]) -> str:
        account = self.backend.get_account("initiate_refund", args["account_id"])
        amount = args["amount"]

        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ToolError(
                ErrorCategory.VALIDATION,
                f"Refund amount must be a positive number; got {amount!r}. "
                "Fix the amount; do not retry with the same value.",
                details={"errorCode": "BAD_AMOUNT", "amount": amount},
            )

        if account.frozen:
            raise ToolError(
                ErrorCategory.PERMISSION,
                f"Account {account.account_id} is frozen; no money movement is "
                "permitted. Do not retry. Escalate with create_support_ticket "
                "(priority=high) so a human can review the freeze.",
                details={"errorCode": "ACCOUNT_FROZEN"},
            )

        txn = self.backend.find_transaction(account, args["transaction_id"])
        original = abs(txn.amount)
        if txn.amount >= 0:
            raise ToolError(
                ErrorCategory.VALIDATION,
                f"Transaction {txn.txn_id} is a credit, not a charge — it cannot "
                "be refunded. Verify the transaction id via get_transaction_history.",
                details={"errorCode": "NOT_A_CHARGE", "transactionId": txn.txn_id},
            )
        if amount > original:
            raise ToolError(
                ErrorCategory.VALIDATION,
                f"Refund amount {amount:.2f} exceeds the original charge of "
                f"{original:.2f} on {txn.txn_id}. Refund at most the original "
                "amount; do not retry with the same value.",
                details={
                    "errorCode": "AMOUNT_EXCEEDS_CHARGE",
                    "maxRefundable": original,
                },
            )

        # NOTE: the threshold policy is enforced by the pre-tool-use hook in
        # hooks.py BEFORE this handler runs. The check here is defense in
        # depth in case the executor is used without the hook installed.
        if amount > self.refund_auto_approval_limit:
            raise ToolError(
                ErrorCategory.PERMISSION,
                f"Refunds above {self.refund_auto_approval_limit:.2f} "
                f"{account.currency} require human approval. Do not retry and do "
                "not split the refund. Escalate with create_support_ticket "
                "(priority=high), including the transaction id and amount.",
                details={
                    "errorCode": "REFUND_LIMIT_EXCEEDED",
                    "limit": self.refund_auto_approval_limit,
                },
            )

        refund_id = f"REFUND-{len(self.backend.refunds) + 1:04d}"
        account.balance += amount
        record = {
            "refundId": refund_id,
            "accountId": account.account_id,
            "transactionId": txn.txn_id,
            "amount": amount,
            "reason": args.get("reason", ""),
            "status": "completed",
        }
        self.backend.refunds.append(record)
        return success_payload(record)

    def _create_support_ticket(self, args: dict[str, Any]) -> str:
        self.backend._maybe_fail("create_support_ticket")
        ticket = {
            "ticketId": self.backend.next_ticket_id(),
            "accountId": args["account_id"],
            "summary": args["summary"],
            "details": args["details"],
            "priority": args.get("priority", "normal"),
            "status": "open",
        }
        self.backend.tickets.append(ticket)
        return success_payload(ticket)
