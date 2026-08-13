/**
 * Programmatic pre-tool-use hooks.
 *
 * A hook runs BEFORE a tool executes and can veto the call by returning a
 * `ToolError`. Returning `null` allows the call to proceed. This is the
 * harness-level enforcement layer: unlike prompt instructions, the model
 * cannot talk its way around a hook — the blocked call never reaches the
 * backend.
 *
 * `RefundThresholdHook` implements the exercise's business rule: refunds
 * above a dollar threshold are intercepted and redirected to the
 * human-escalation workflow (create_support_ticket).
 */

import { ToolError } from "./errors.js";
import type { ToolInput } from "./tools.js";

export interface PreToolUseHook {
  /** Return a ToolError to block the call, or null to allow it. */
  check(toolName: string, toolInput: ToolInput): ToolError | null;
}

/** Block initiate_refund calls above `threshold` and redirect to escalation. */
export class RefundThresholdHook implements PreToolUseHook {
  readonly blockedCalls: ToolInput[] = [];

  constructor(readonly threshold = 500.0) {}

  check(toolName: string, toolInput: ToolInput): ToolError | null {
    if (toolName !== "initiate_refund") return null;
    const amount = toolInput.amount;
    if (typeof amount === "number" && amount > this.threshold) {
      this.blockedCalls.push({ ...toolInput });
      return new ToolError(
        "permission",
        `BLOCKED BY POLICY: refunds above ${this.threshold.toFixed(2)} require ` +
          "human approval, so this call was intercepted before execution. " +
          "Do not retry it and do not split the amount into smaller refunds. " +
          "Instead, escalate: call create_support_ticket with priority='high', " +
          "including the account id, transaction id, requested amount " +
          `(${amount.toFixed(2)}), and the user's reason. Then tell the user ` +
          "their refund request was forwarded for human approval.",
        {
          errorCode: "HOOK_REFUND_LIMIT",
          limit: this.threshold,
          requestedAmount: amount,
          escalationTool: "create_support_ticket",
        },
      );
    }
    return null;
  }
}

/** Ordered collection of pre-tool-use hooks; first veto wins. */
export class HookRegistry {
  private readonly hooks: PreToolUseHook[];

  constructor(hooks: PreToolUseHook[] = []) {
    this.hooks = [...hooks];
  }

  add(hook: PreToolUseHook): void {
    this.hooks.push(hook);
  }

  check(toolName: string, toolInput: ToolInput): ToolError | null {
    for (const hook of this.hooks) {
      const veto = hook.check(toolName, toolInput);
      if (veto) return veto;
    }
    return null;
  }
}
