"""The agentic loop.

Drives request -> stop_reason check -> tool execution -> tool_result -> repeat,
until the model finishes ("end_turn") or a safety limit is hit.

stop_reason handling:

- "tool_use"   -> run hooks, execute every tool_use block, send ALL results
                  back in a single user message (parallel calls stay parallel)
- "end_turn"   -> the model is done; return the final text
- "pause_turn" -> server-side pause; re-send the turn to let it resume
- "max_tokens" -> output truncated; surface an error rather than pretending
                  the answer is complete
- "refusal"    -> safety refusal; surface it (never blindly retried)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .errors import ToolError
from .hooks import HookRegistry
from .tools import TOOL_DEFINITIONS, ToolExecutor

logger = logging.getLogger("agentic_loop")

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You are a banking support agent for AnyBank. You help customers check "
    "balances, review transactions, issue refunds, and escalate to human "
    "support when policy requires it.\n\n"
    "Tool-error handling rules:\n"
    "- Every failed tool call returns JSON with errorCategory and isRetryable.\n"
    "- errorCategory=transient (isRetryable=true): retry the same call. If it "
    "still fails after 3 attempts, tell the user the service is down and offer "
    "to open a support ticket.\n"
    "- errorCategory=validation (isRetryable=false): never repeat the identical "
    "call. Fix the input if you can, or explain the business rule to the user "
    "in plain language.\n"
    "- errorCategory=permission (isRetryable=false): never retry and never work "
    "around the block. Follow the escalation guidance in the error message "
    "(usually create_support_ticket), then tell the user what happened.\n\n"
    "When a request contains several independent concerns, address every one "
    "of them: use the tools each concern needs (in parallel where possible) and "
    "finish with a single unified answer that covers all concerns explicitly."
)


@dataclass
class LoopResult:
    final_text: str
    stop_reason: str
    turns: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    def __init__(
        self,
        client: Any,
        executor: ToolExecutor,
        hooks: Optional[HookRegistry] = None,
        model: str = DEFAULT_MODEL,
        system: str = SYSTEM_PROMPT,
        max_turns: int = 15,
    ):
        self.client = client
        self.executor = executor
        self.hooks = hooks or HookRegistry()
        self.model = model
        self.system = system
        self.max_turns = max_turns

    def run(self, user_message: str) -> LoopResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        tool_calls: list[dict[str, Any]] = []

        for turn in range(1, self.max_turns + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=self.system,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            stop_reason = response.stop_reason
            logger.info("turn %d: stop_reason=%s", turn, stop_reason)

            if stop_reason == "tool_use":
                # Echo the assistant turn (including tool_use blocks) into
                # history, then execute every requested tool.
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    result_block = self._run_tool(block, tool_calls)
                    tool_results.append(result_block)
                # All results for the turn go back in ONE user message.
                messages.append({"role": "user", "content": tool_results})
                continue

            if stop_reason == "pause_turn":
                # Server paused mid-turn; append and re-send to resume.
                messages.append({"role": "assistant", "content": response.content})
                continue

            if stop_reason == "end_turn":
                return LoopResult(
                    final_text=self._extract_text(response),
                    stop_reason=stop_reason,
                    turns=turn,
                    tool_calls=tool_calls,
                    messages=messages,
                )

            if stop_reason == "max_tokens":
                raise RuntimeError(
                    "Response truncated at max_tokens; increase the limit or stream."
                )

            if stop_reason == "refusal":
                return LoopResult(
                    final_text=(
                        "The request was declined by safety systems and cannot be "
                        "completed."
                    ),
                    stop_reason=stop_reason,
                    turns=turn,
                    tool_calls=tool_calls,
                    messages=messages,
                )

            raise RuntimeError(f"Unhandled stop_reason: {stop_reason!r}")

        raise RuntimeError(f"Agent did not finish within {self.max_turns} turns.")

    # -- internals --------------------------------------------------------------

    def _run_tool(self, block: Any, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        """Hook check + execution for one tool_use block -> tool_result block."""
        name, tool_input = block.name, dict(block.input)
        record: dict[str, Any] = {"tool": name, "input": tool_input}
        tool_calls.append(record)

        veto = self.hooks.check(name, tool_input)
        if veto is not None:
            logger.info("hook blocked %s: %s", name, veto.details.get("errorCode"))
            record.update(status="blocked_by_hook", error=veto.to_payload())
            return self._error_result(block.id, veto)

        try:
            output = self.executor.execute(name, tool_input)
        except ToolError as err:
            logger.info("tool %s failed: %s", name, err.category.value)
            record.update(status="error", error=err.to_payload())
            return self._error_result(block.id, err)

        record.update(status="ok", output=json.loads(output))
        return {"type": "tool_result", "tool_use_id": block.id, "content": output}

    @staticmethod
    def _error_result(tool_use_id: str, err: ToolError) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": err.to_json(),
            "is_error": True,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        return "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
