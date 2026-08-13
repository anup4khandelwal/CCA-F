/**
 * Structured tool error model.
 *
 * Every tool failure is reported to the model as a JSON object with three
 * required fields:
 *
 * - `errorCategory`: one of `transient` / `validation` / `permission`
 * - `isRetryable`:   whether calling the tool again with the same input can
 *                    reasonably be expected to succeed
 * - `message`:       a human-readable description written *for the model*,
 *                    including guidance about what to do next
 *
 * The categories map to distinct agent behaviors:
 *
 * - `transient`  -> retry the same call (isRetryable = true)
 * - `validation` -> do NOT retry; fix the input or explain the business rule
 *                   to the user (isRetryable = false)
 * - `permission` -> do NOT retry; follow the escalation guidance in the
 *                   message (isRetryable = false)
 */

export type ErrorCategory = "transient" | "validation" | "permission";

export interface ToolErrorPayload {
  ok: false;
  errorCategory: ErrorCategory;
  isRetryable: boolean;
  message: string;
  details: Record<string, unknown>;
}

/** Thrown by tool implementations; serialized into the tool_result. */
export class ToolError extends Error {
  constructor(
    readonly category: ErrorCategory,
    message: string,
    readonly details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ToolError";
  }

  get isRetryable(): boolean {
    return this.category === "transient";
  }

  toPayload(): ToolErrorPayload {
    return {
      ok: false,
      errorCategory: this.category,
      isRetryable: this.isRetryable,
      message: this.message,
      details: this.details,
    };
  }

  toJson(): string {
    return JSON.stringify(this.toPayload());
  }
}

/** Serialize a successful tool result. */
export function successPayload(data: Record<string, unknown>): string {
  return JSON.stringify({ ok: true, ...data });
}
