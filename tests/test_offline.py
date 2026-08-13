"""Offline tests for the agentic loop, hooks, and structured errors.

These use a scripted FakeClient that plays back canned model turns, so the
loop's stop_reason branching, hook enforcement, and error propagation are all
verified without network access or an API key. Live end-to-end scenarios
(where the real model decides what to call) live in run_live.py.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_loop.backend import BankBackend
from agentic_loop.errors import ErrorCategory, ToolError
from agentic_loop.hooks import HookRegistry, RefundThresholdHook
from agentic_loop.loop import AgentLoop
from agentic_loop.tools import TOOL_DEFINITIONS, ToolExecutor


# ---------------------------------------------------------------------------
# Scripted fake client
# ---------------------------------------------------------------------------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(block_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeClient:
    """Plays back a scripted sequence of responses and records requests."""

    def __init__(self, script):
        self._script = list(script)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._script:
            raise AssertionError("FakeClient script exhausted — loop did not stop")
        return self._script.pop(0)


def make_loop(script, threshold=500.0, backend=None):
    backend = backend or BankBackend()
    executor = ToolExecutor(backend, refund_auto_approval_limit=threshold)
    hooks = HookRegistry([RefundThresholdHook(threshold=threshold)])
    client = FakeClient(script)
    loop = AgentLoop(client, executor, hooks=hooks, model="fake-model")
    return loop, client, executor, backend


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class ToolDefinitionTests(unittest.TestCase):
    def test_four_tools_with_schemas(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        self.assertEqual(
            names,
            [
                "get_account_balance",
                "get_transaction_history",
                "initiate_refund",
                "create_support_ticket",
            ],
        )
        for tool in TOOL_DEFINITIONS:
            self.assertGreater(
                len(tool["description"]), 200, f"{tool['name']} description too thin"
            )
            self.assertEqual(tool["input_schema"]["type"], "object")
            self.assertTrue(tool["input_schema"]["required"])

    def test_similar_tools_state_boundaries(self):
        """The two overlapping read tools must cross-reference each other."""
        by_name = {t["name"]: t["description"] for t in TOOL_DEFINITIONS}
        self.assertIn("get_transaction_history", by_name["get_account_balance"])
        self.assertIn("get_account_balance", by_name["get_transaction_history"])


# ---------------------------------------------------------------------------
# stop_reason branching
# ---------------------------------------------------------------------------

class StopReasonTests(unittest.TestCase):
    def test_end_turn_returns_final_text(self):
        loop, client, _, _ = make_loop(
            [response("end_turn", [text_block("Hello! How can I help?")])]
        )
        result = loop.run("hi")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.final_text, "Hello! How can I help?")
        self.assertEqual(result.turns, 1)
        self.assertEqual(len(client.requests), 1)

    def test_tool_use_then_end_turn(self):
        loop, client, executor, _ = make_loop(
            [
                response(
                    "tool_use",
                    [
                        text_block("Checking your balance."),
                        tool_use_block(
                            "tu_1", "get_account_balance", {"account_id": "ACC-1001"}
                        ),
                    ],
                ),
                response("end_turn", [text_block("Your balance is $2450.75.")]),
            ]
        )
        result = loop.run("What's my balance? Account ACC-1001.")
        self.assertEqual(result.final_text, "Your balance is $2450.75.")
        self.assertEqual(executor.call_log[0][0], "get_account_balance")

        # Second request must carry assistant echo + tool_result user message.
        second = client.requests[1]["messages"]
        self.assertEqual(second[1]["role"], "assistant")
        self.assertEqual(second[2]["role"], "user")
        tool_result = second[2]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "tu_1")
        payload = json.loads(tool_result["content"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["balance"], 2450.75)

    def test_pause_turn_resumes(self):
        loop, client, _, _ = make_loop(
            [
                response("pause_turn", [text_block("...working...")]),
                response("end_turn", [text_block("Done.")]),
            ]
        )
        result = loop.run("long task")
        self.assertEqual(result.final_text, "Done.")
        self.assertEqual(len(client.requests), 2)

    def test_refusal_is_surfaced_not_retried(self):
        loop, client, _, _ = make_loop([response("refusal", [])])
        result = loop.run("bad request")
        self.assertEqual(result.stop_reason, "refusal")
        self.assertEqual(len(client.requests), 1)

    def test_max_tokens_raises(self):
        loop, _, _, _ = make_loop([response("max_tokens", [text_block("truncat")])])
        with self.assertRaises(RuntimeError):
            loop.run("hi")


# ---------------------------------------------------------------------------
# Structured errors
# ---------------------------------------------------------------------------

class StructuredErrorTests(unittest.TestCase):
    def test_transient_error_is_retryable_and_retry_succeeds(self):
        backend = BankBackend()
        backend.inject_transient_failures("get_transaction_history", 1)
        script = [
            response(
                "tool_use",
                [tool_use_block("tu_1", "get_transaction_history", {"account_id": "ACC-1001"})],
            ),
            # The scripted "model" reads the transient error and retries.
            response(
                "tool_use",
                [tool_use_block("tu_2", "get_transaction_history", {"account_id": "ACC-1001"})],
            ),
            response("end_turn", [text_block("Here are your transactions.")]),
        ]
        loop, client, executor, _ = make_loop(script, backend=backend)
        result = loop.run("show my history")

        first_error = json.loads(client.requests[1]["messages"][2]["content"][0]["content"])
        self.assertFalse(first_error["ok"])
        self.assertEqual(first_error["errorCategory"], "transient")
        self.assertTrue(first_error["isRetryable"])
        self.assertTrue(client.requests[1]["messages"][2]["content"][0]["is_error"])

        second_result = json.loads(client.requests[2]["messages"][4]["content"][0]["content"])
        self.assertTrue(second_result["ok"])
        self.assertEqual(len(executor.call_log), 2)
        self.assertEqual(result.stop_reason, "end_turn")

    def test_validation_error_not_retryable(self):
        script = [
            response(
                "tool_use",
                [
                    tool_use_block(
                        "tu_1",
                        "initiate_refund",
                        {
                            "account_id": "ACC-1001",
                            "transaction_id": "TXN-9002",
                            "amount": -50,
                            "reason": "test",
                        },
                    )
                ],
            ),
            response("end_turn", [text_block("That amount isn't valid.")]),
        ]
        loop, client, _, backend = make_loop(script)
        loop.run("refund me -50")
        err = json.loads(client.requests[1]["messages"][2]["content"][0]["content"])
        self.assertEqual(err["errorCategory"], "validation")
        self.assertFalse(err["isRetryable"])
        self.assertEqual(backend.refunds, [])

    def test_permission_error_on_frozen_account(self):
        script = [
            response(
                "tool_use",
                [
                    tool_use_block(
                        "tu_1",
                        "initiate_refund",
                        {
                            "account_id": "ACC-2002",
                            "transaction_id": "TXN-9101",
                            "amount": 8.50,
                            "reason": "bad coffee",
                        },
                    )
                ],
            ),
            response("end_turn", [text_block("Your account is frozen; escalating.")]),
        ]
        loop, client, _, _ = make_loop(script)
        loop.run("refund my coffee")
        err = json.loads(client.requests[1]["messages"][2]["content"][0]["content"])
        self.assertEqual(err["errorCategory"], "permission")
        self.assertFalse(err["isRetryable"])
        self.assertIn("create_support_ticket", err["message"])

    def test_tool_error_payload_shape(self):
        err = ToolError(ErrorCategory.TRANSIENT, "boom", {"errorCode": "X"})
        payload = err.to_payload()
        self.assertEqual(
            set(payload), {"ok", "errorCategory", "isRetryable", "message", "details"}
        )
        self.assertTrue(payload["isRetryable"])


# ---------------------------------------------------------------------------
# Hook enforcement + escalation
# ---------------------------------------------------------------------------

class HookTests(unittest.TestCase):
    def test_hook_blocks_over_threshold_and_tool_never_executes(self):
        script = [
            response(
                "tool_use",
                [
                    tool_use_block(
                        "tu_1",
                        "initiate_refund",
                        {
                            "account_id": "ACC-1001",
                            "transaction_id": "TXN-9001",
                            "amount": 750.0,
                            "reason": "disputed charge",
                        },
                    )
                ],
            ),
            # Scripted model follows the escalation guidance.
            response(
                "tool_use",
                [
                    tool_use_block(
                        "tu_2",
                        "create_support_ticket",
                        {
                            "account_id": "ACC-1001",
                            "summary": "Refund over auto-approval limit",
                            "details": "Refund of 750.00 on TXN-9001 requires human approval.",
                            "priority": "high",
                        },
                    )
                ],
            ),
            response("end_turn", [text_block("Escalated to a human for approval.")]),
        ]
        loop, client, executor, backend = make_loop(script, threshold=500.0)
        result = loop.run("refund $750 from TXN-9001")

        # Hook fired before execution: initiate_refund never hit the executor.
        executed = [name for name, _ in executor.call_log]
        self.assertNotIn("initiate_refund", executed)
        self.assertEqual(backend.refunds, [])

        blocked = json.loads(client.requests[1]["messages"][2]["content"][0]["content"])
        self.assertEqual(blocked["errorCategory"], "permission")
        self.assertEqual(blocked["details"]["errorCode"], "HOOK_REFUND_LIMIT")
        self.assertEqual(blocked["details"]["escalationTool"], "create_support_ticket")

        # Escalation completed.
        self.assertEqual(len(backend.tickets), 1)
        self.assertEqual(backend.tickets[0]["priority"], "high")
        self.assertEqual(result.stop_reason, "end_turn")

    def test_hook_allows_under_threshold(self):
        hook = RefundThresholdHook(threshold=500.0)
        self.assertIsNone(
            hook("initiate_refund", {"amount": 19.99, "account_id": "ACC-1001"})
        )
        self.assertIsNone(hook("get_account_balance", {"account_id": "ACC-1001"}))
        veto = hook("initiate_refund", {"amount": 500.01})
        self.assertIsNotNone(veto)
        self.assertEqual(veto.category, ErrorCategory.PERMISSION)
        self.assertEqual(len(hook.blocked_calls), 1)


# ---------------------------------------------------------------------------
# Multi-concern decomposition
# ---------------------------------------------------------------------------

class MultiConcernTests(unittest.TestCase):
    def test_parallel_tool_calls_return_in_one_user_message(self):
        """Three concerns -> three tool_use blocks in one assistant turn.

        The loop must execute all three and send all three tool_results back
        in a SINGLE user message (splitting them degrades parallel calling).
        """
        script = [
            response(
                "tool_use",
                [
                    tool_use_block("tu_1", "get_account_balance", {"account_id": "ACC-1001"}),
                    tool_use_block(
                        "tu_2",
                        "get_transaction_history",
                        {"account_id": "ACC-1001", "start_date": "2026-08-01", "end_date": "2026-08-05"},
                    ),
                    tool_use_block(
                        "tu_3",
                        "initiate_refund",
                        {
                            "account_id": "ACC-1001",
                            "transaction_id": "TXN-9003",
                            "amount": 19.99,
                            "reason": "duplicate subscription charge",
                        },
                    ),
                ],
            ),
            response(
                "end_turn",
                [
                    text_block(
                        "1) Balance: $2470.74. 2) Your Aug 1 utility payment went "
                        "through. 3) Refunded the duplicate $19.99 charge."
                    )
                ],
            ),
        ]
        loop, client, executor, backend = make_loop(script)
        result = loop.run(
            "I was double-charged $19.99 by Streamify — refund the duplicate. "
            "Also, what's my balance, and did my utility payment on Aug 1 go "
            "through? Account ACC-1001."
        )

        # All three tools executed.
        self.assertEqual(
            [name for name, _ in executor.call_log],
            ["get_account_balance", "get_transaction_history", "initiate_refund"],
        )
        # All three results in one user message, ids matched pairwise.
        results_msg = client.requests[1]["messages"][2]
        self.assertEqual(results_msg["role"], "user")
        self.assertEqual(
            [r["tool_use_id"] for r in results_msg["content"]], ["tu_1", "tu_2", "tu_3"]
        )
        self.assertTrue(all(r["type"] == "tool_result" for r in results_msg["content"]))
        # Refund actually applied.
        self.assertEqual(len(backend.refunds), 1)
        self.assertAlmostEqual(backend.accounts["ACC-1001"].balance, 2470.74)
        # Unified answer covers all three concerns.
        self.assertEqual(result.tool_calls[2]["status"], "ok")
        self.assertIn("Refunded", result.final_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
