/**
 * Fake banking backend used by the tools.
 *
 * Deterministic in-memory data plus a configurable transient-failure injector
 * so tests and demos can exercise the `transient` error path predictably.
 */

import { ToolError } from "./errors.js";

export interface Transaction {
  txnId: string;
  date: string; // ISO date
  description: string;
  amount: number; // negative = charge, positive = credit
}

export interface Account {
  accountId: string;
  owner: string;
  balance: number;
  currency: string;
  frozen: boolean;
  transactions: Transaction[];
}

export interface Ticket {
  ticketId: string;
  accountId: string;
  summary: string;
  details: string;
  priority: "low" | "normal" | "high";
  status: "open";
}

export interface Refund {
  refundId: string;
  accountId: string;
  transactionId: string;
  amount: number;
  reason: string;
  status: "completed";
}

function seedAccounts(): Map<string, Account> {
  return new Map<string, Account>([
    [
      "ACC-1001",
      {
        accountId: "ACC-1001",
        owner: "Jordan Lee",
        balance: 2450.75,
        currency: "USD",
        frozen: false,
        transactions: [
          { txnId: "TXN-9001", date: "2026-08-01", description: "Utility bill payment", amount: -120.0 },
          { txnId: "TXN-9002", date: "2026-08-03", description: "Streamify subscription", amount: -19.99 },
          { txnId: "TXN-9003", date: "2026-08-03", description: "Streamify subscription", amount: -19.99 },
          { txnId: "TXN-9004", date: "2026-08-07", description: "Salary deposit", amount: 3200.0 },
          { txnId: "TXN-9005", date: "2026-08-10", description: "Grocery store", amount: -86.42 },
        ],
      },
    ],
    [
      "ACC-2002",
      {
        accountId: "ACC-2002",
        owner: "Sam Rivera",
        balance: 110.1,
        currency: "USD",
        frozen: true,
        transactions: [
          { txnId: "TXN-9101", date: "2026-08-05", description: "Cafe purchase", amount: -8.5 },
        ],
      },
    ],
  ]);
}

/**
 * In-memory backend with per-tool transient failure injection.
 *
 * `injectTransientFailures(toolName, n)` makes the next *n* calls that tool
 * routes through this backend throw a retryable transient error before
 * succeeding — simulating flaky downstream services.
 */
export class BankBackend {
  readonly accounts = seedAccounts();
  readonly tickets: Ticket[] = [];
  readonly refunds: Refund[] = [];
  private pendingFailures = new Map<string, number>();
  private ticketSeq = 5000;

  // -- failure injection ----------------------------------------------------

  injectTransientFailures(toolName: string, count: number): void {
    this.pendingFailures.set(toolName, count);
  }

  maybeFail(toolName: string): void {
    const remaining = this.pendingFailures.get(toolName) ?? 0;
    if (remaining > 0) {
      this.pendingFailures.set(toolName, remaining - 1);
      throw new ToolError(
        "transient",
        "The banking service is temporarily unavailable (upstream timeout). " +
          "This error is transient — retry the same tool call.",
        { errorCode: "SERVICE_UNAVAILABLE", retryAfterSeconds: 1 },
      );
    }
  }

  // -- data access ------------------------------------------------------------

  getAccount(toolName: string, accountId: string): Account {
    this.maybeFail(toolName);
    const account = this.accounts.get(accountId);
    if (!account) {
      throw new ToolError(
        "validation",
        `No account exists with id '${accountId}'. Do not retry with the same ` +
          "id; ask the user to confirm their account id (format: ACC-NNNN).",
        { errorCode: "ACCOUNT_NOT_FOUND", accountId },
      );
    }
    return account;
  }

  findTransaction(account: Account, txnId: string): Transaction {
    const txn = account.transactions.find((t) => t.txnId === txnId);
    if (!txn) {
      throw new ToolError(
        "validation",
        `Transaction '${txnId}' does not exist on account ` +
          `'${account.accountId}'. Do not retry; verify the transaction id ` +
          "via get_transaction_history first.",
        { errorCode: "TXN_NOT_FOUND", transactionId: txnId },
      );
    }
    return txn;
  }

  nextTicketId(): string {
    this.ticketSeq += 1;
    return `TICKET-${this.ticketSeq}`;
  }
}
