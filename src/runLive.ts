/**
 * Live end-to-end scenarios against the real Claude API.
 *
 * Requires ANTHROPIC_API_KEY in the environment:
 *
 *     export ANTHROPIC_API_KEY=sk-ant-...
 *     npm run live              # run all scenarios
 *     npm run live -- hook      # run one scenario by name
 *
 * Each scenario prints the tool-call trace (including hook blocks and
 * structured errors) and the model's final answer, then asserts the behavior
 * the exercise asks for.
 */

import assert from "node:assert";

import Anthropic from "@anthropic-ai/sdk";

import { BankBackend } from "./backend.js";
import { HookRegistry, RefundThresholdHook } from "./hooks.js";
import { AgentLoop, type LoopResult } from "./loop.js";
import { ToolExecutor } from "./tools.js";

function buildLoop(threshold = 500.0) {
  const backend = new BankBackend();
  const executor = new ToolExecutor(backend, threshold);
  const hook = new RefundThresholdHook(threshold);
  const loop = new AgentLoop(new Anthropic(), executor, {
    hooks: new HookRegistry([hook]),
    log: (line) => console.log(`  # ${line}`),
  });
  return { loop, backend, executor, hook };
}

function show(result: LoopResult): void {
  console.log("\n--- tool calls ---");
  for (const call of result.toolCalls) {
    let line = `  [${call.status ?? "?"}] ${call.tool}(${JSON.stringify(call.input)})`;
    if (call.error) {
      line += ` -> ${call.error.errorCategory} (retryable=${call.error.isRetryable})`;
    }
    console.log(line);
  }
  console.log("--- final answer ---");
  console.log(result.finalText);
  console.log(`--- (${result.turns} turns, stop_reason=${result.stopReason}) ---\n`);
}

/** Baseline: single tool call, end_turn. */
async function scenarioSimple(): Promise<void> {
  const { loop, executor } = buildLoop();
  const result = await loop.run("What's my current balance? My account is ACC-1001.");
  show(result);
  assert(executor.callLog.some((c) => c.toolName === "get_account_balance"));
}

/** The overlapping tools: a past-activity question must pick history, not balance. */
async function scenarioToolSelection(): Promise<void> {
  const { loop, executor } = buildLoop();
  const result = await loop.run(
    "Did my utility bill payment on August 1st go through? Account ACC-1001.",
  );
  show(result);
  const called = executor.callLog.map((c) => c.toolName);
  assert(called.includes("get_transaction_history"), called.join(", "));
}

/** First two history calls fail transiently; the agent should retry to success. */
async function scenarioTransientRetry(): Promise<void> {
  const { loop, backend, executor } = buildLoop();
  backend.injectTransientFailures("get_transaction_history", 2);
  const result = await loop.run(
    "Show me my recent transactions for account ACC-1001.",
  );
  show(result);
  const historyCalls = executor.callLog.filter(
    (c) => c.toolName === "get_transaction_history",
  );
  assert(
    historyCalls.length >= 3,
    `expected retries, got ${historyCalls.length} call(s)`,
  );
}

/** Refund larger than the original charge -> validation error, no blind retry. */
async function scenarioValidationError(): Promise<void> {
  const { loop, backend } = buildLoop();
  const result = await loop.run(
    "Refund me $100 for the Streamify charge TXN-9002 on account ACC-1001.",
  );
  show(result);
  // The $19.99 charge can't yield a $100 refund; agent should either refund
  // at most 19.99 (after explaining) or explain the rule — never a 100 refund.
  assert(
    backend.refunds.every((r) => r.amount <= 19.99),
    JSON.stringify(backend.refunds),
  );
}

/** Refund above the $500 threshold -> hook blocks -> support ticket created. */
async function scenarioHookEscalation(): Promise<void> {
  const { loop, backend, hook } = buildLoop(500.0);
  const result = await loop.run(
    "Please issue a $750 refund on account ACC-1001 for transaction " +
      "TXN-9001 (reason: billing dispute).",
  );
  show(result);
  assert(hook.blockedCalls.length > 0, "expected the hook to intercept the refund");
  assert.deepStrictEqual(backend.refunds, []);
  assert(backend.tickets.length > 0, "expected an escalation ticket");
}

/** Three concerns in one message: refund + balance + payment status. */
async function scenarioMultiConcern(): Promise<void> {
  const { loop, backend, executor } = buildLoop();
  const result = await loop.run(
    "Hi — three things on account ACC-1001: (1) I think Streamify charged " +
      "me twice for $19.99 this month, please refund the duplicate; (2) what's " +
      "my current balance; (3) did my utility bill payment on August 1st go " +
      "through?",
  );
  show(result);
  const called = new Set(executor.callLog.map((c) => c.toolName));
  assert(called.has("get_account_balance"));
  assert(called.has("get_transaction_history"));
  assert(backend.refunds.length === 1 && backend.refunds[0].amount === 19.99);
}

const SCENARIOS: Record<string, () => Promise<void>> = {
  simple: scenarioSimple,
  selection: scenarioToolSelection,
  transient: scenarioTransientRetry,
  validation: scenarioValidationError,
  hook: scenarioHookEscalation,
  multi: scenarioMultiConcern,
};

async function main(): Promise<void> {
  const names = process.argv.slice(2);
  const toRun = names.length > 0 ? names : Object.keys(SCENARIOS);
  for (const name of toRun) {
    const scenario = SCENARIOS[name];
    if (!scenario) throw new Error(`Unknown scenario '${name}'`);
    console.log(`\n================ scenario: ${name} ================`);
    await scenario();
  }
  console.log("All scenarios passed.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
