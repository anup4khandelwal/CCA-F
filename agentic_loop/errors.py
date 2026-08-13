"""Structured tool error model.

Every tool failure is reported to the model as a JSON object with three
required fields:

- ``errorCategory``: one of ``transient`` / ``validation`` / ``permission``
- ``isRetryable``:   whether calling the tool again with the same input can
                     reasonably be expected to succeed
- ``message``:       a human-readable description written *for the model*,
                     including guidance about what to do next

The categories map to distinct agent behaviors:

- ``transient``  -> retry the same call (isRetryable = True)
- ``validation`` -> do NOT retry; fix the input or explain the business rule
                    to the user (isRetryable = False)
- ``permission`` -> do NOT retry; follow the escalation guidance in the
                    message (isRetryable = False)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    VALIDATION = "validation"
    PERMISSION = "permission"


@dataclass
class ToolError(Exception):
    """Raised by tool implementations; serialized into the tool_result."""

    category: ErrorCategory
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_retryable(self) -> bool:
        return self.category is ErrorCategory.TRANSIENT

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": False,
            "errorCategory": self.category.value,
            "isRetryable": self.is_retryable,
            "message": self.message,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True)


def success_payload(data: dict[str, Any]) -> str:
    """Serialize a successful tool result."""
    return json.dumps({"ok": True, **data}, sort_keys=True, default=str)
