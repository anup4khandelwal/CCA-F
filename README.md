# Agentic Loop with Tool Integration, Structured Errors & Hooks

A practice project covering agentic architecture (Domain 1), tool design / MCP
integration (Domain 2), and context management & reliability (Domain 5). It
implements a small banking-support agent on the Anthropic Messages API
(TypeScript, `@anthropic-ai/sdk`) with a hand-written agentic loop, MCP-style
tool contracts, structured error handling, and a programmatic policy hook with
an escalation workflow.

## Layout

```
src/
  tools.ts       # 4 tool definitions (MCP-shaped: name + description + JSON Schema)
                 #   two deliberately-overlapping read tools with boundary-drawing
                 #   descriptions, plus refund + escalation tools
  loop.ts        # the agentic loop: branches on stop_reason
                 #   (tool_use / end_turn / pause_turn / max_tokens / refusal)
  errors.ts      # structured errors: errorCategory + isRetryable + message
  hooks.ts       # pre-tool-use hook layer; RefundThresholdHook blocks refunds
                 #   > $500 and redirects to the create_support_ticket escalation
  backend.ts     # deterministic in-memory bank + transient-failure injector
  mcpServer.ts   # the same tools served over a real MCP stdio server
  runLive.ts     # end-to-end scenarios against the real API (needs your key)
tests/
  loop.test.ts   # loop/hook/error tests with a scripted fake client (no API key)
```

## How each exercise requirement is met

**1. Tool design with boundary conditions.** `get_account_balance` and
`get_transaction_history` overlap (both read account data), so each
description states exactly what it does *and does not* return, and names the
other tool for the out-of-scope case ("did my payment go through" → history,
never balance). Every description also documents its error contract. The
definitions use the MCP tool shape (name / description / inputSchema) and
`mcpServer.ts` serves them over real MCP via `@modelcontextprotocol/sdk`.

**2. stop_reason-driven loop.** `loop.ts` requests → checks `stop_reason`:
`tool_use` executes every tool block and returns *all* results in a single
user message (keeps parallel calling healthy), `end_turn` returns the final
text, `pause_turn` re-sends to resume, `max_tokens` and `refusal` are
surfaced rather than silently swallowed.

**3. Structured errors.** Every failure is a JSON payload
`{ok:false, errorCategory, isRetryable, message, details}` sent back as a
`tool_result` with `is_error:true`:

| category   | isRetryable | expected agent behavior                          |
|------------|-------------|--------------------------------------------------|
| transient  | true        | retry the same call (bounded by system prompt)   |
| validation | false       | fix input or explain the business rule to user   |
| permission | false       | never retry; follow escalation guidance          |

The system prompt tells the model how to treat each category; the error
`message` fields repeat the guidance in-context.

**4. Programmatic hook + escalation.** `RefundThresholdHook` intercepts
`initiate_refund` calls **before execution** — a blocked call never reaches
the backend. The veto is returned to the model as a `permission` error whose
message directs it to `create_support_ticket` (priority high), producing the
escalation workflow. The executor keeps a duplicate threshold check as
defense in depth.

**5. Multi-concern messages.** The system prompt instructs decomposition; the
loop supports parallel `tool_use` blocks; the `multi` scenario in
`src/runLive.ts` (and the multi-concern test offline) verify a 3-concern
message triggers all three tools and one synthesized answer.

## Running

```bash
npm install
npm run typecheck        # strict TS, no emit
npm test                 # 14 offline tests via node:test — no API key needed
```

Live scenarios (uses model `claude-opus-5`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
npm run live                    # all six scenarios
npm run live -- hook            # just the hook/escalation scenario
npm run live -- transient multi
```

Scenarios: `simple` (baseline single call), `selection` (overlapping-tool
disambiguation), `transient` (injected failures → retries), `validation`
(over-refund rejected), `hook` ($750 refund blocked → ticket), `multi`
(three concerns in one message).

MCP server:

```bash
npm run mcp     # serves the same 4 tools over stdio
```
