/**
 * Tool definitions and implementations.
 *
 * The four tools are written MCP-style: each definition is a name + detailed
 * description + JSON Schema, exactly the shape an MCP server advertises via
 * `tools/list` (and the same shape the Anthropic Messages API accepts).
 * `mcpServer.ts` exposes these same tools over a real MCP transport.
 *
 * Two tools deliberately overlap — `get_account_balance` and
 * `get_transaction_history` both read account data — so their descriptions
 * must draw a sharp boundary to prevent the model from picking the wrong one:
 *
 * - balance  = a single point-in-time number, no transaction detail
 * - history  = a list of dated transactions, never the current balance
 */

import type Anthropic from "@anthropic-ai/sdk";

import { BankBackend } from "./backend.js";
import { successPayload, ToolError } from "./errors.js";

export type ToolInput = Record<string, unknown>;

export const TOOL_DEFINITIONS: Anthropic.Tool[] = [
  {
    name: "get_account_balance",
    description:
      "Return the CURRENT balance of one account as a single point-in-time " +
      "number, with currency and account status. Use this when the user asks " +
      "'how much money do I have', 'what's my balance', or you need to check " +
      "available funds before a refund.\n\n" +
      "Boundaries: this tool returns ONLY the current balance snapshot — it " +
      "never returns individual transactions, payment history, or whether a " +
      "specific payment went through. For any question about past activity " +
      "('did my payment go through', 'what did I spend last week', 'was I " +
      "double-charged') use get_transaction_history instead.\n\n" +
      "Errors: returns a structured error with errorCategory=validation if the " +
      "account id is unknown (do not retry — confirm the id with the user), or " +
      "errorCategory=transient if the banking service times out (retry the " +
      "same call).",
    input_schema: {
      type: "object",
      properties: {
        account_id: {
          type: "string",
          description: "Account identifier, format ACC-NNNN, e.g. ACC-1001.",
        },
      },
      required: ["account_id"],
    },
  },
  {
    name: "get_transaction_history",
    description:
      "Return the LIST of individual transactions on an account — each with " +
      "transaction id, date, description, and signed amount (negative = " +
      "charge, positive = credit) — optionally filtered to a date range. Use " +
      "this to answer any question about PAST ACTIVITY: 'did my payment on " +
      "Aug 1 go through', 'was I charged twice', 'show my recent purchases', " +
      "or to look up a transaction id before initiating a refund.\n\n" +
      "Boundaries: this tool never returns the current account balance — the " +
      "amounts are per-transaction, not a running total. For 'how much money " +
      "is in my account right now' use get_account_balance instead. This tool " +
      "is read-only; it cannot reverse or refund a charge — use " +
      "initiate_refund for that.\n\n" +
      "Errors: errorCategory=validation for an unknown account id or a " +
      "malformed date (do not retry with the same input); " +
      "errorCategory=transient for upstream timeouts (retry the same call).",
    input_schema: {
      type: "object",
      properties: {
        account_id: {
          type: "string",
          description: "Account identifier, format ACC-NNNN.",
        },
        start_date: {
          type: "string",
          description: "Optional inclusive ISO start date (YYYY-MM-DD).",
        },
        end_date: {
          type: "string",
          description: "Optional inclusive ISO end date (YYYY-MM-DD).",
        },
      },
      required: ["account_id"],
    },
  },
  {
    name: "initiate_refund",
    description:
      "Reverse a specific past charge by issuing a refund to the account. " +
      "Requires the exact transaction id of the original charge (look it up " +
      "with get_transaction_history first if the user only describes the " +
      "charge). The refund amount may be at most the absolute value of the " +
      "original charge.\n\n" +
      "Business rules enforced by this tool and the surrounding platform:\n" +
      "- amount must be a positive number no greater than the original charge\n" +
      "- the account must not be frozen\n" +
      "- refunds above the auto-approval threshold are BLOCKED by policy and " +
      "return errorCategory=permission with instructions to escalate via " +
      "create_support_ticket. Do not retry a blocked refund and do not split " +
      "it into smaller refunds to evade the threshold — escalate instead.\n\n" +
      "Errors: errorCategory=validation for bad amounts or unknown " +
      "transaction ids (fix the input, don't retry blindly); " +
      "errorCategory=permission for policy blocks (escalate, never retry); " +
      "errorCategory=transient for service timeouts (retry the same call).",
    input_schema: {
      type: "object",
      properties: {
        account_id: {
          type: "string",
          description: "Account identifier, format ACC-NNNN.",
        },
        transaction_id: {
          type: "string",
          description: "Id of the original charge to refund, e.g. TXN-9002.",
        },
        amount: {
          type: "number",
          description:
            "Refund amount in the account currency. Positive, and at most " +
            "the absolute value of the original charge.",
        },
        reason: {
          type: "string",
          description: "Short reason for the refund, e.g. 'duplicate charge'.",
        },
      },
      required: ["account_id", "transaction_id", "amount", "reason"],
    },
  },
  {
    name: "create_support_ticket",
    description:
      "Create a ticket for the human support team. This is the ESCALATION " +
      "path: use it when an action is blocked by policy (e.g. a refund above " +
      "the auto-approval threshold returned errorCategory=permission), when " +
      "an account is frozen, or when the user asks for something no other " +
      "tool can do. It does not move money or change account state itself — " +
      "it queues the request for a human.\n\n" +
      "Include everything a human agent needs to act without re-asking the " +
      "user: the account id, relevant transaction ids and amounts, and what " +
      "the user wants. Set priority='high' only for blocked money movements " +
      "or frozen-account issues.\n\n" +
      "Errors: errorCategory=transient if the ticketing service is briefly " +
      "unavailable (retry the same call).",
    input_schema: {
      type: "object",
      properties: {
        account_id: {
          type: "string",
          description: "Account identifier the ticket concerns.",
        },
        summary: {
          type: "string",
          description: "One-line summary of the request.",
        },
        details: {
          type: "string",
          description:
            "Full context for the human agent: transaction ids, amounts, " +
            "what was attempted, and why it was escalated.",
        },
        priority: {
          type: "string",
          enum: ["low", "normal", "high"],
          description: "Ticket priority. Default normal.",
        },
      },
      required: ["account_id", "summary", "details"],
    },
  },
];

export interface ToolCallLogEntry {
  toolName: string;
  input: ToolInput;
}

/**
 * Dispatches validated tool calls against the backend.
 *
 * Returns a JSON string on success and throws ToolError on failure; the
 * agentic loop converts either into a tool_result block.
 */
export class ToolExecutor {
  /** Observable call log for tests / demos. */
  readonly callLog: ToolCallLogEntry[] = [];

  constructor(
    readonly backend: BankBackend,
    readonly refundAutoApprovalLimit = 500.0,
  ) {}

  execute(toolName: string, input: ToolInput): string {
    this.callLog.push({ toolName, input: { ...input } });
    switch (toolName) {
      case "get_account_balance":
        return this.getAccountBalance(input);
      case "get_transaction_history":
        return this.getTransactionHistory(input);
      case "initiate_refund":
        return this.initiateRefund(input);
      case "create_support_ticket":
        return this.createSupportTicket(input);
      default:
        throw new ToolError("validation", `Unknown tool '${toolName}'. Do not retry.`, {
          errorCode: "UNKNOWN_TOOL",
        });
    }
  }

  // -- handlers ----------------------------------------------------------------

  private getAccountBalance(args: ToolInput): string {
    const account = this.backend.getAccount("get_account_balance", String(args.account_id));
    return successPayload({
      accountId: account.accountId,
      balance: Math.round(account.balance * 100) / 100,
      currency: account.currency,
      status: account.frozen ? "frozen" : "active",
    });
  }

  private getTransactionHistory(args: ToolInput): string {
    const account = this.backend.getAccount(
      "get_transaction_history",
      String(args.account_id),
    );
    const start = args.start_date as string | undefined;
    const end = args.end_date as string | undefined;
    for (const [label, value] of [
      ["start_date", start],
      ["end_date", end],
    ] as const) {
      if (value !== undefined && !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
        throw new ToolError(
          "validation",
          `${label} '${value}' is not a valid ISO date (YYYY-MM-DD). ` +
            "Fix the date format; do not retry with the same value.",
          { errorCode: "BAD_DATE", field: label },
        );
      }
    }
    const transactions = account.transactions.filter(
      (t) => (start === undefined || t.date >= start) && (end === undefined || t.date <= end),
    );
    return successPayload({
      accountId: account.accountId,
      transactions,
      count: transactions.length,
    });
  }

  private initiateRefund(args: ToolInput): string {
    const account = this.backend.getAccount("initiate_refund", String(args.account_id));
    const amount = args.amount;

    if (typeof amount !== "number" || !Number.isFinite(amount) || amount <= 0) {
      throw new ToolError(
        "validation",
        `Refund amount must be a positive number; got ${JSON.stringify(amount)}. ` +
          "Fix the amount; do not retry with the same value.",
        { errorCode: "BAD_AMOUNT", amount },
      );
    }

    if (account.frozen) {
      throw new ToolError(
        "permission",
        `Account ${account.accountId} is frozen; no money movement is ` +
          "permitted. Do not retry. Escalate with create_support_ticket " +
          "(priority=high) so a human can review the freeze.",
        { errorCode: "ACCOUNT_FROZEN" },
      );
    }

    const txn = this.backend.findTransaction(account, String(args.transaction_id));
    const original = Math.abs(txn.amount);
    if (txn.amount >= 0) {
      throw new ToolError(
        "validation",
        `Transaction ${txn.txnId} is a credit, not a charge — it cannot ` +
          "be refunded. Verify the transaction id via get_transaction_history.",
        { errorCode: "NOT_A_CHARGE", transactionId: txn.txnId },
      );
    }
    if (amount > original) {
      throw new ToolError(
        "validation",
        `Refund amount ${amount.toFixed(2)} exceeds the original charge of ` +
          `${original.toFixed(2)} on ${txn.txnId}. Refund at most the original ` +
          "amount; do not retry with the same value.",
        { errorCode: "AMOUNT_EXCEEDS_CHARGE", maxRefundable: original },
      );
    }

    // NOTE: the threshold policy is enforced by the pre-tool-use hook in
    // hooks.ts BEFORE this handler runs. The check here is defense in depth
    // in case the executor is used without the hook installed.
    if (amount > this.refundAutoApprovalLimit) {
      throw new ToolError(
        "permission",
        `Refunds above ${this.refundAutoApprovalLimit.toFixed(2)} ` +
          `${account.currency} require human approval. Do not retry and do ` +
          "not split the refund. Escalate with create_support_ticket " +
          "(priority=high), including the transaction id and amount.",
        { errorCode: "REFUND_LIMIT_EXCEEDED", limit: this.refundAutoApprovalLimit },
      );
    }

    const refundId = `REFUND-${String(this.backend.refunds.length + 1).padStart(4, "0")}`;
    account.balance += amount;
    const record = {
      refundId,
      accountId: account.accountId,
      transactionId: txn.txnId,
      amount,
      reason: String(args.reason ?? ""),
      status: "completed" as const,
    };
    this.backend.refunds.push(record);
    return successPayload(record);
  }

  private createSupportTicket(args: ToolInput): string {
    this.backend.maybeFail("create_support_ticket");
    const priority = (args.priority as "low" | "normal" | "high" | undefined) ?? "normal";
    const ticket = {
      ticketId: this.backend.nextTicketId(),
      accountId: String(args.account_id),
      summary: String(args.summary),
      details: String(args.details),
      priority,
      status: "open" as const,
    };
    this.backend.tickets.push(ticket);
    return successPayload(ticket);
  }
}
