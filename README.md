# Agentic Loop with Tool Integration, Structured Errors & Hooks

A practice project covering agentic architecture (Domain 1), tool design / MCP
integration (Domain 2), and context management & reliability (Domain 5). It
implements a small banking-support agent on the Anthropic Messages API with a
hand-written agentic loop, MCP-style tool contracts, structured error
handling, and a programmatic policy hook with an escalation workflow.

## Layout

```
agentic_loop/
  tools.py       # 4 tool definitions (MCP-shaped: name + description + JSON Schema)
                 #   two deliberately-overlapping read tools with boundary-drawing
                 #   descriptions, plus refund + escalation tools
  loop.py        # the agentic loop: branches on stop_reason
                 #   (tool_use / end_turn / pause_turn / max_tokens / refusal)
  errors.py      # structured errors: errorCategory + isRetryable + message
  hooks.py       # pre-tool-use hook layer; RefundThresholdHook blocks refunds
                 #   > $500 and redirects to the create_support_ticket escalation
  backend.py     # deterministic in-memory bank + transient-failure injector
  mcp_server.py  # optional: the same tools served over a real MCP stdio server
tests/
  test_offline.py  # loop/hook/error tests with a scripted fake client (no API key)
run_live.py        # end-to-end scenarios against the real API (needs your key)
```

## How each exercise requirement is met

**1. Tool design with boundary conditions.** `get_account_balance` and
`get_transaction_history` overlap (both read account data), so each
description states exactly what it does *and does not* return, and names the
other tool for the out-of-scope case ("did my payment go through" → history,
never balance). Every description also documents its error contract. The
definitions use the MCP tool shape (name / description / inputSchema) and
`mcp_server.py` serves them over real MCP.

**2. stop_reason-driven loop.** `loop.py` requests → checks `stop_reason`:
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
loop supports parallel `tool_use` blocks; `scenario_multi` in `run_live.py`
(and `MultiConcernTests` offline) verify a 3-concern message triggers all
three tools and one synthesized answer.

## Running

Offline tests (no key needed):

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Live scenarios (uses model `claude-opus-5`):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python run_live.py            # all six scenarios
python run_live.py hook       # just the hook/escalation scenario
python run_live.py transient multi
```

Scenarios: `simple` (baseline single call), `selection` (overlapping-tool
disambiguation), `transient` (injected failures → retries), `validation`
(over-refund rejected), `hook` ($750 refund blocked → ticket), `multi`
(three concerns in one message).

Optional MCP server (`pip install mcp`):

```bash
python -m agentic_loop.mcp_server   # serves the same 4 tools over stdio
```
