/**
 * Offline tests for the agentic loop, hooks, and structured errors.
 *
 * These use a scripted FakeClient that plays back canned model turns, so the
 * loop's stop_reason branching, hook enforcement, and error propagation are
 * all verified without network access or an API key. Live end-to-end
 * scenarios (where the real model decides what to call) live in
 * src/runLive.ts.
 *
 * Run: npm test
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type Anthropic from "@anthropic-ai/sdk";

import { BankBackend } from "../src/backend.js";
import { ToolError, type ToolErrorPayload } from "../src/errors.js";
import { HookRegistry, RefundThresholdHook } from "../src/hooks.js";
import { AgentLoop, type MessagesClient } from "../src/loop.js";
import { TOOL_DEFINITIONS, ToolExecutor, type ToolInput } from "../src/tools.js";

// ---------------------------------------------------------------------------
// Scripted fake client
// ---------------------------------------------------------------------------

function textBlock(text: string) {
  return { type: "text", text } as const;
}

function toolUseBlock(id: string, name: string, input: ToolInput) {
  return { type: "tool_use", id, name, input } as const;
}

function response(stopReason: string, content: unknown[]): Anthropic.Message {
  return { stop_reason: stopReason, content } as unknown as Anthropic.Message;
}

/** Plays back a scripted sequence of responses and records requests. */
class FakeClient implements MessagesClient {
  readonly requests: Anthropic.MessageCreateParamsNonStreaming[] = [];
  readonly messages = {
    create: async (
      params: Anthropic.MessageCreateParamsNonStreaming,
    ): Promise<Anthropic.Message> => {
      this.requests.push(params);
      const next = this.script.shift();
      assert(next, "FakeClient script exhausted — loop did not stop");
      return next;
    },
  };

  constructor(private readonly script: Anthropic.Message[]) {}
}

function makeLoop(
  script: Anthropic.Message[],
  options: { threshold?: number; backend?: BankBackend } = {},
) {
  const threshold = options.threshold ?? 500.0;
  const backend = options.backend ?? new BankBackend();
  const executor = new ToolExecutor(backend, threshold);
  const client = new FakeClient(script);
  const loop = new AgentLoop(client, executor, {
    hooks: new HookRegistry([new RefundThresholdHook(threshold)]),
    model: "fake-model",
  });
  return { loop, client, executor, backend };
}

function contentBlocks(
  message: Anthropic.MessageParam,
): Anthropic.ToolResultBlockParam[] {
  return message.content as Anthropic.ToolResultBlockParam[];
}

function errorPayload(block: Anthropic.ToolResultBlockParam): ToolErrorPayload {
  return JSON.parse(block.content as string) as ToolErrorPayload;
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

describe("tool definitions", () => {
  it("declares four tools with schemas and substantial descriptions", () => {
    assert.deepEqual(
      TOOL_DEFINITIONS.map((t) => t.name),
      [
        "get_account_balance",
        "get_transaction_history",
        "initiate_refund",
        "create_support_ticket",
      ],
    );
    for (const tool of TOOL_DEFINITIONS) {
      assert(
        (tool.description ?? "").length > 200,
        `${tool.name} description too thin`,
      );
      assert.equal(tool.input_schema.type, "object");
      assert((tool.input_schema.required as string[]).length > 0);
    }
  });

  it("overlapping read tools cross-reference each other's boundaries", () => {
    const byName = new Map(TOOL_DEFINITIONS.map((t) => [t.name, t.description ?? ""]));
    assert(byName.get("get_account_balance")!.includes("get_transaction_history"));
    assert(byName.get("get_transaction_history")!.includes("get_account_balance"));
  });
});

// ---------------------------------------------------------------------------
// stop_reason branching
// ---------------------------------------------------------------------------

describe("stop_reason branching", () => {
  it("end_turn returns the final text", async () => {
    const { loop, client } = makeLoop([
      response("end_turn", [textBlock("Hello! How can I help?")]),
    ]);
    const result = await loop.run("hi");
    assert.equal(result.stopReason, "end_turn");
    assert.equal(result.finalText, "Hello! How can I help?");
    assert.equal(result.turns, 1);
    assert.equal(client.requests.length, 1);
  });

  it("tool_use executes the tool and feeds the result back", async () => {
    const { loop, client, executor } = makeLoop([
      response("tool_use", [
        textBlock("Checking your balance."),
        toolUseBlock("tu_1", "get_account_balance", { account_id: "ACC-1001" }),
      ]),
      response("end_turn", [textBlock("Your balance is $2450.75.")]),
    ]);
    const result = await loop.run("What's my balance? Account ACC-1001.");
    assert.equal(result.finalText, "Your balance is $2450.75.");
    assert.equal(executor.callLog[0].toolName, "get_account_balance");

    // Second request must carry assistant echo + tool_result user message.
    const second = client.requests[1].messages;
    assert.equal(second[1].role, "assistant");
    assert.equal(second[2].role, "user");
    const toolResult = contentBlocks(second[2])[0];
    assert.equal(toolResult.type, "tool_result");
    assert.equal(toolResult.tool_use_id, "tu_1");
    const payload = JSON.parse(toolResult.content as string);
    assert.equal(payload.ok, true);
    assert.equal(payload.balance, 2450.75);
  });

  it("pause_turn re-sends to resume", async () => {
    const { loop, client } = makeLoop([
      response("pause_turn", [textBlock("...working...")]),
      response("end_turn", [textBlock("Done.")]),
    ]);
    const result = await loop.run("long task");
    assert.equal(result.finalText, "Done.");
    assert.equal(client.requests.length, 2);
  });

  it("refusal is surfaced, not retried", async () => {
    const { loop, client } = makeLoop([response("refusal", [])]);
    const result = await loop.run("bad request");
    assert.equal(result.stopReason, "refusal");
    assert.equal(client.requests.length, 1);
  });

  it("max_tokens raises instead of returning a truncated answer", async () => {
    const { loop } = makeLoop([response("max_tokens", [textBlock("truncat")])]);
    await assert.rejects(loop.run("hi"), /max_tokens/);
  });
});

// ---------------------------------------------------------------------------
// Structured errors
// ---------------------------------------------------------------------------

describe("structured errors", () => {
  it("transient error is retryable and the retry succeeds", async () => {
    const backend = new BankBackend();
    backend.injectTransientFailures("get_transaction_history", 1);
    const { loop, client, executor } = makeLoop(
      [
        response("tool_use", [
          toolUseBlock("tu_1", "get_transaction_history", { account_id: "ACC-1001" }),
        ]),
        // The scripted "model" reads the transient error and retries.
        response("tool_use", [
          toolUseBlock("tu_2", "get_transaction_history", { account_id: "ACC-1001" }),
        ]),
        response("end_turn", [textBlock("Here are your transactions.")]),
      ],
      { backend },
    );
    const result = await loop.run("show my history");

    const firstResult = contentBlocks(client.requests[1].messages[2])[0];
    const firstError = errorPayload(firstResult);
    assert.equal(firstError.ok, false);
    assert.equal(firstError.errorCategory, "transient");
    assert.equal(firstError.isRetryable, true);
    assert.equal(firstResult.is_error, true);

    const secondPayload = JSON.parse(
      contentBlocks(client.requests[2].messages[4])[0].content as string,
    );
    assert.equal(secondPayload.ok, true);
    assert.equal(executor.callLog.length, 2);
    assert.equal(result.stopReason, "end_turn");
  });

  it("validation error is not retryable and no refund happens", async () => {
    const { loop, client, backend } = makeLoop([
      response("tool_use", [
        toolUseBlock("tu_1", "initiate_refund", {
          account_id: "ACC-1001",
          transaction_id: "TXN-9002",
          amount: -50,
          reason: "test",
        }),
      ]),
      response("end_turn", [textBlock("That amount isn't valid.")]),
    ]);
    await loop.run("refund me -50");
    const err = errorPayload(contentBlocks(client.requests[1].messages[2])[0]);
    assert.equal(err.errorCategory, "validation");
    assert.equal(err.isRetryable, false);
    assert.deepEqual(backend.refunds, []);
  });

  it("permission error on a frozen account carries escalation guidance", async () => {
    const { loop, client } = makeLoop([
      response("tool_use", [
        toolUseBlock("tu_1", "initiate_refund", {
          account_id: "ACC-2002",
          transaction_id: "TXN-9101",
          amount: 8.5,
          reason: "bad coffee",
        }),
      ]),
      response("end_turn", [textBlock("Your account is frozen; escalating.")]),
    ]);
    await loop.run("refund my coffee");
    const err = errorPayload(contentBlocks(client.requests[1].messages[2])[0]);
    assert.equal(err.errorCategory, "permission");
    assert.equal(err.isRetryable, false);
    assert(err.message.includes("create_support_ticket"));
  });

  it("ToolError payload has the required shape", () => {
    const err = new ToolError("transient", "boom", { errorCode: "X" });
    const payload = err.toPayload();
    assert.deepEqual(
      Object.keys(payload).sort(),
      ["details", "errorCategory", "isRetryable", "message", "ok"],
    );
    assert.equal(payload.isRetryable, true);
  });
});

// ---------------------------------------------------------------------------
// Hook enforcement + escalation
// ---------------------------------------------------------------------------

describe("pre-tool-use hooks", () => {
  it("blocks refunds over the threshold before execution and escalates", async () => {
    const { loop, client, executor, backend } = makeLoop(
      [
        response("tool_use", [
          toolUseBlock("tu_1", "initiate_refund", {
            account_id: "ACC-1001",
            transaction_id: "TXN-9001",
            amount: 750.0,
            reason: "disputed charge",
          }),
        ]),
        // Scripted model follows the escalation guidance.
        response("tool_use", [
          toolUseBlock("tu_2", "create_support_ticket", {
            account_id: "ACC-1001",
            summary: "Refund over auto-approval limit",
            details: "Refund of 750.00 on TXN-9001 requires human approval.",
            priority: "high",
          }),
        ]),
        response("end_turn", [textBlock("Escalated to a human for approval.")]),
      ],
      { threshold: 500.0 },
    );
    const result = await loop.run("refund $750 from TXN-9001");

    // Hook fired before execution: initiate_refund never hit the executor.
    const executed = executor.callLog.map((c) => c.toolName);
    assert(!executed.includes("initiate_refund"));
    assert.deepEqual(backend.refunds, []);

    const blocked = errorPayload(contentBlocks(client.requests[1].messages[2])[0]);
    assert.equal(blocked.errorCategory, "permission");
    assert.equal(blocked.details.errorCode, "HOOK_REFUND_LIMIT");
    assert.equal(blocked.details.escalationTool, "create_support_ticket");

    // Escalation completed.
    assert.equal(backend.tickets.length, 1);
    assert.equal(backend.tickets[0].priority, "high");
    assert.equal(result.stopReason, "end_turn");
  });

  it("allows calls under the threshold and untouched tools", () => {
    const hook = new RefundThresholdHook(500.0);
    assert.equal(hook.check("initiate_refund", { amount: 19.99, account_id: "ACC-1001" }), null);
    assert.equal(hook.check("get_account_balance", { account_id: "ACC-1001" }), null);
    const veto = hook.check("initiate_refund", { amount: 500.01 });
    assert(veto instanceof ToolError);
    assert.equal(veto.category, "permission");
    assert.equal(hook.blockedCalls.length, 1);
  });
});

// ---------------------------------------------------------------------------
// Multi-concern decomposition
// ---------------------------------------------------------------------------

describe("multi-concern messages", () => {
  it("executes parallel tool calls and batches all results in one user message", async () => {
    // Three concerns -> three tool_use blocks in one assistant turn. The
    // loop must execute all three and send all three tool_results back in a
    // SINGLE user message (splitting them degrades parallel calling).
    const { loop, client, executor, backend } = makeLoop([
      response("tool_use", [
        toolUseBlock("tu_1", "get_account_balance", { account_id: "ACC-1001" }),
        toolUseBlock("tu_2", "get_transaction_history", {
          account_id: "ACC-1001",
          start_date: "2026-08-01",
          end_date: "2026-08-05",
        }),
        toolUseBlock("tu_3", "initiate_refund", {
          account_id: "ACC-1001",
          transaction_id: "TXN-9003",
          amount: 19.99,
          reason: "duplicate subscription charge",
        }),
      ]),
      response("end_turn", [
        textBlock(
          "1) Balance: $2470.74. 2) Your Aug 1 utility payment went " +
            "through. 3) Refunded the duplicate $19.99 charge.",
        ),
      ]),
    ]);
    const result = await loop.run(
      "I was double-charged $19.99 by Streamify — refund the duplicate. " +
        "Also, what's my balance, and did my utility payment on Aug 1 go " +
        "through? Account ACC-1001.",
    );

    // All three tools executed.
    assert.deepEqual(
      executor.callLog.map((c) => c.toolName),
      ["get_account_balance", "get_transaction_history", "initiate_refund"],
    );
    // All three results in one user message, ids matched pairwise.
    const resultsMsg = client.requests[1].messages[2];
    assert.equal(resultsMsg.role, "user");
    assert.deepEqual(
      contentBlocks(resultsMsg).map((r) => r.tool_use_id),
      ["tu_1", "tu_2", "tu_3"],
    );
    assert(contentBlocks(resultsMsg).every((r) => r.type === "tool_result"));
    // Refund actually applied.
    assert.equal(backend.refunds.length, 1);
    assert.ok(Math.abs(backend.accounts.get("ACC-1001")!.balance - 2470.74) < 1e-9);
    // Unified answer covers all three concerns.
    assert.equal(result.toolCalls[2].status, "ok");
    assert(result.finalText.includes("Refunded"));
  });
});
