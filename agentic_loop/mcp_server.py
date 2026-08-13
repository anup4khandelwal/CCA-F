"""Expose the four tools over a real MCP server (optional).

Requires the ``mcp`` package (``pip install mcp``). Run with:

    python -m agentic_loop.mcp_server

An MCP client (Claude Desktop, Claude Code, or the Anthropic API's MCP
connector via a URL transport) then discovers the same tool names, detailed
descriptions, and JSON Schemas defined in tools.py — demonstrating that the
tool contract is transport-independent: the agentic loop in loop.py consumes
it directly, and MCP clients consume it via tools/list + tools/call.

Structured errors survive the transport: a ToolError is serialized to the
same {ok, errorCategory, isRetryable, message, details} JSON and flagged
isError=True in the MCP tool result.
"""

from __future__ import annotations

import anyio

from .backend import BankBackend
from .errors import ToolError
from .tools import TOOL_DEFINITIONS, ToolExecutor


def build_server():
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    backend = BankBackend()
    executor = ToolExecutor(backend)
    server = Server("anybank-support")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["input_schema"],
            )
            for t in TOOL_DEFINITIONS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            output = executor.execute(name, arguments)
        except ToolError as err:
            # MCP signals failure via isError on the result; the structured
            # payload rides in the content, same as the direct-API path.
            raise RuntimeError(err.to_json()) from err
        return [TextContent(type="text", text=output)]

    return server


async def main() -> None:
    from mcp.server.stdio import stdio_server

    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(main)
