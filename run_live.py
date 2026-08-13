"""Live end-to-end scenarios against the real Claude API.

Requires ANTHROPIC_API_KEY in the environment:

    export ANTHROPIC_API_KEY=sk-ant-...
    python run_live.py            # run all scenarios
    python run_live.py hook       # run one scenario by name

Each scenario prints the tool-call trace (including hook blocks and
structured errors) and the model's final answer, then asserts the behavior
the exercise asks for.
"""

from __future__ import annotations

import logging
import sys

import anthropic

from agentic_loop.backend import BankBackend
from agentic_loop.hooks import HookRegistry, RefundThresholdHook
from agentic_loop.loop import AgentLoop
from agentic_loop.tools import ToolExecutor

logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_loop(threshold: float = 500.0):
    backend = BankBackend()
    executor = ToolExecutor(backend, refund_auto_approval_limit=threshold)
    hook = RefundThresholdHook(threshold=threshold)
    loop = AgentLoop(
        anthropic.Anthropic(),
        executor,
        hooks=HookRegistry([hook]),
    )
    return loop, backend, executor, hook


def show(result):
    print("\n--- tool calls ---")
    for call in result.tool_calls:
        status = call.get("status", "?")
        line = f"  [{status}] {call['tool']}({call['input']})"
        if "error" in call:
            err = call["error"]
            line += f" -> {err['errorCategory']} (retryable={err['isRetryable']})"
        print(line)
    print("--- final answer ---")
    print(result.final_text)
    print(f"--- ({result.turns} turns, stop_reason={result.stop_reason}) ---\n")


def scenario_simple():
    """Baseline: single tool call, end_turn."""
    loop, _, executor, _ = build_loop()
    result = loop.run("What's my current balance? My account is ACC-1001.")
    show(result)
    assert any(name == "get_account_balance" for name, _ in executor.call_log)


def scenario_tool_selection():
    """The overlapping tools: a past-activity question must pick history, not balance."""
    loop, _, executor, _ = build_loop()
    result = loop.run(
        "Did my utility bill payment on August 1st go through? Account ACC-1001."
    )
    show(result)
    called = [name for name, _ in executor.call_log]
    assert "get_transaction_history" in called, called


def scenario_transient_retry():
    """First two history calls fail transiently; the agent should retry to success."""
    loop, backend, executor, _ = build_loop()
    backend.inject_transient_failures("get_transaction_history", 2)
    result = loop.run("Show me my recent transactions for account ACC-1001.")
    show(result)
    history_calls = [n for n, _ in executor.call_log if n == "get_transaction_history"]
    assert len(history_calls) >= 3, f"expected retries, got {len(history_calls)} call(s)"


def scenario_validation_error():
    """Refund larger than the original charge -> validation error, no blind retry."""
    loop, backend, _, _ = build_loop()
    result = loop.run(
        "Refund me $100 for the Streamify charge TXN-9002 on account ACC-1001."
    )
    show(result)
    # The $19.99 charge can't yield a $100 refund; agent should either refund
    # at most 19.99 (after explaining) or explain the rule — never a 100 refund.
    assert all(r["amount"] <= 19.99 for r in backend.refunds), backend.refunds


def scenario_hook_escalation():
    """Refund above the $500 threshold -> hook blocks -> support ticket created."""
    loop, backend, _, hook = build_loop(threshold=500.0)
    result = loop.run(
        "Please issue a $750 refund on account ACC-1001 for transaction "
        "TXN-9001 (reason: billing dispute)."
    )
    show(result)
    assert hook.blocked_calls, "expected the hook to intercept the refund"
    assert backend.refunds == []
    assert backend.tickets, "expected an escalation ticket"


def scenario_multi_concern():
    """Three concerns in one message: refund + balance + payment status."""
    loop, backend, executor, _ = build_loop()
    result = loop.run(
        "Hi — three things on account ACC-1001: (1) I think Streamify charged "
        "me twice for $19.99 this month, please refund the duplicate; (2) what's "
        "my current balance; (3) did my utility bill payment on August 1st go "
        "through?"
    )
    show(result)
    called = {name for name, _ in executor.call_log}
    assert "get_account_balance" in called
    assert "get_transaction_history" in called
    assert len(backend.refunds) == 1 and backend.refunds[0]["amount"] == 19.99


SCENARIOS = {
    "simple": scenario_simple,
    "selection": scenario_tool_selection,
    "transient": scenario_transient_retry,
    "validation": scenario_validation_error,
    "hook": scenario_hook_escalation,
    "multi": scenario_multi_concern,
}


def main() -> None:
    names = sys.argv[1:] or list(SCENARIOS)
    for name in names:
        print(f"\n================ scenario: {name} ================")
        SCENARIOS[name]()
    print("All scenarios passed.")


if __name__ == "__main__":
    main()
