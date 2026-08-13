"""Programmatic pre-tool-use hooks.

A hook runs BEFORE a tool executes and can veto the call by returning a
``ToolError``. Returning ``None`` allows the call to proceed. This is the
harness-level enforcement layer: unlike prompt instructions, the model cannot
talk its way around a hook — the blocked call never reaches the backend.

``RefundThresholdHook`` implements the exercise's business rule: refunds above
a dollar threshold are intercepted and redirected to the human-escalation
workflow (create_support_ticket).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .errors import ErrorCategory, ToolError


class PreToolUseHook(Protocol):
    def __call__(self, tool_name: str, tool_input: dict[str, Any]) -> Optional[ToolError]:
        """Return a ToolError to block the call, or None to allow it."""
        ...


@dataclass
class RefundThresholdHook:
    """Block initiate_refund calls above ``threshold`` and redirect to escalation."""

    threshold: float = 500.0
    blocked_calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, tool_name: str, tool_input: dict[str, Any]) -> Optional[ToolError]:
        if tool_name != "initiate_refund":
            return None
        amount = tool_input.get("amount")
        if isinstance(amount, (int, float)) and amount > self.threshold:
            self.blocked_calls.append(dict(tool_input))
            return ToolError(
                ErrorCategory.PERMISSION,
                f"BLOCKED BY POLICY: refunds above {self.threshold:.2f} require "
                "human approval, so this call was intercepted before execution. "
                "Do not retry it and do not split the amount into smaller refunds. "
                "Instead, escalate: call create_support_ticket with priority='high', "
                "including the account id, transaction id, requested amount "
                f"({amount:.2f}), and the user's reason. Then tell the user their "
                "refund request was forwarded for human approval.",
                details={
                    "errorCode": "HOOK_REFUND_LIMIT",
                    "limit": self.threshold,
                    "requestedAmount": amount,
                    "escalationTool": "create_support_ticket",
                },
            )
        return None


class HookRegistry:
    """Ordered collection of pre-tool-use hooks; first veto wins."""

    def __init__(self, hooks: Optional[list[PreToolUseHook]] = None):
        self.hooks: list[PreToolUseHook] = list(hooks or [])

    def add(self, hook: PreToolUseHook) -> None:
        self.hooks.append(hook)

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> Optional[ToolError]:
        for hook in self.hooks:
            veto = hook(tool_name, tool_input)
            if veto is not None:
                return veto
        return None
