/**
 * The agentic loop.
 *
 * Drives request -> stop_reason check -> tool execution -> tool_result ->
 * repeat, until the model finishes ("end_turn") or a safety limit is hit.
 *
 * stop_reason handling:
 *
 * - "tool_use"   -> run hooks, execute every tool_use block, send ALL results
 *                   back in a single user message (parallel calls stay parallel)
 * - "end_turn"   -> the model is done; return the final text
 * - "pause_turn" -> server-side pause; re-send the turn to let it resume
 * - "max_tokens" -> output truncated; surface an error rather than pretending
 *                   the answer is complete
 * - "refusal"    -> safety refusal; surface it (never blindly retried)
 */

import type Anthropic from "@anthropic-ai/sdk";

import { ToolError, type ToolErrorPayload } from "./errors.js";
import { HookRegistry } from "./hooks.js";
import { TOOL_DEFINITIONS, ToolExecutor, type ToolInput } from "./tools.js";

export const DEFAULT_MODEL = "claude-opus-5";

export const SYSTEM_PROMPT =
  "You are a banking support agent for AnyBank. You help customers check " +
  "balances, review transactions, issue refunds, and escalate to human " +
  "support when policy requires it.\n\n" +
  "Tool-error handling rules:\n" +
  "- Every failed tool call returns JSON with errorCategory and isRetryable.\n" +
  "- errorCategory=transient (isRetryable=true): retry the same call. If it " +
  "still fails after 3 attempts, tell the user the service is down and offer " +
  "to open a support ticket.\n" +
  "- errorCategory=validation (isRetryable=false): never repeat the identical " +
  "call. Fix the input if you can, or explain the business rule to the user " +
  "in plain language.\n" +
  "- errorCategory=permission (isRetryable=false): never retry and never work " +
  "around the block. Follow the escalation guidance in the error message " +
  "(usually create_support_ticket), then tell the user what happened.\n\n" +
  "When a request contains several independent concerns, address every one " +
  "of them: use the tools each concern needs (in parallel where possible) and " +
  "finish with a single unified answer that covers all concerns explicitly.";

/**
 * The minimal client surface the loop needs. The real Anthropic client
 * satisfies it structurally; tests substitute a scripted fake.
 */
export interface MessagesClient {
  messages: {
    create(
      params: Anthropic.MessageCreateParamsNonStreaming,
    ): Promise<Anthropic.Message>;
  };
}

export interface ToolCallRecord {
  tool: string;
  input: ToolInput;
  status?: "ok" | "error" | "blocked_by_hook";
  error?: ToolErrorPayload;
  output?: unknown;
}

export interface LoopResult {
  finalText: string;
  stopReason: string;
  turns: number;
  toolCalls: ToolCallRecord[];
  messages: Anthropic.MessageParam[];
}

export interface AgentLoopOptions {
  hooks?: HookRegistry;
  model?: string;
  system?: string;
  maxTurns?: number;
  log?: (line: string) => void;
}

export class AgentLoop {
  private readonly hooks: HookRegistry;
  private readonly model: string;
  private readonly system: string;
  private readonly maxTurns: number;
  private readonly log: (line: string) => void;

  constructor(
    private readonly client: MessagesClient,
    private readonly executor: ToolExecutor,
    options: AgentLoopOptions = {},
  ) {
    this.hooks = options.hooks ?? new HookRegistry();
    this.model = options.model ?? DEFAULT_MODEL;
    this.system = options.system ?? SYSTEM_PROMPT;
    this.maxTurns = options.maxTurns ?? 15;
    this.log = options.log ?? (() => {});
  }

  async run(userMessage: string): Promise<LoopResult> {
    const messages: Anthropic.MessageParam[] = [{ role: "user", content: userMessage }];
    const toolCalls: ToolCallRecord[] = [];

    for (let turn = 1; turn <= this.maxTurns; turn++) {
      const response = await this.client.messages.create({
        model: this.model,
        max_tokens: 16000,
        system: this.system,
        tools: TOOL_DEFINITIONS,
        messages,
      });
      const stopReason = response.stop_reason;
      this.log(`turn ${turn}: stop_reason=${stopReason}`);

      if (stopReason === "tool_use") {
        // Echo the assistant turn (including tool_use blocks) into history,
        // then execute every requested tool.
        messages.push({ role: "assistant", content: response.content });
        const toolResults: Anthropic.ToolResultBlockParam[] = [];
        for (const block of response.content) {
          if (block.type !== "tool_use") continue;
          toolResults.push(this.runTool(block, toolCalls));
        }
        // All results for the turn go back in ONE user message.
        messages.push({ role: "user", content: toolResults });
        continue;
      }

      if (stopReason === "pause_turn") {
        // Server paused mid-turn; append and re-send to resume.
        messages.push({ role: "assistant", content: response.content });
        continue;
      }

      if (stopReason === "end_turn") {
        return {
          finalText: extractText(response),
          stopReason,
          turns: turn,
          toolCalls,
          messages,
        };
      }

      if (stopReason === "max_tokens") {
        throw new Error(
          "Response truncated at max_tokens; increase the limit or stream.",
        );
      }

      if (stopReason === "refusal") {
        return {
          finalText:
            "The request was declined by safety systems and cannot be completed.",
          stopReason,
          turns: turn,
          toolCalls,
          messages,
        };
      }

      throw new Error(`Unhandled stop_reason: ${JSON.stringify(stopReason)}`);
    }

    throw new Error(`Agent did not finish within ${this.maxTurns} turns.`);
  }

  // -- internals ------------------------------------------------------------

  /** Hook check + execution for one tool_use block -> tool_result block. */
  private runTool(
    block: Anthropic.ToolUseBlock,
    toolCalls: ToolCallRecord[],
  ): Anthropic.ToolResultBlockParam {
    const input = { ...(block.input as ToolInput) };
    const record: ToolCallRecord = { tool: block.name, input };
    toolCalls.push(record);

    const veto = this.hooks.check(block.name, input);
    if (veto) {
      this.log(`hook blocked ${block.name}: ${String(veto.details.errorCode)}`);
      record.status = "blocked_by_hook";
      record.error = veto.toPayload();
      return errorResult(block.id, veto);
    }

    let output: string;
    try {
      output = this.executor.execute(block.name, input);
    } catch (err) {
      if (err instanceof ToolError) {
        this.log(`tool ${block.name} failed: ${err.category}`);
        record.status = "error";
        record.error = err.toPayload();
        return errorResult(block.id, err);
      }
      throw err;
    }

    record.status = "ok";
    record.output = JSON.parse(output);
    return { type: "tool_result", tool_use_id: block.id, content: output };
  }
}

function errorResult(
  toolUseId: string,
  err: ToolError,
): Anthropic.ToolResultBlockParam {
  return {
    type: "tool_result",
    tool_use_id: toolUseId,
    content: err.toJson(),
    is_error: true,
  };
}

function extractText(response: Anthropic.Message): string {
  return response.content
    .filter((block): block is Anthropic.TextBlock => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}
