/**
 * Expose the four tools over a real MCP server (optional).
 *
 * Run with:
 *
 *     npm run mcp
 *
 * An MCP client (Claude Desktop, Claude Code, or the Anthropic API's MCP
 * connector via a URL transport) then discovers the same tool names, detailed
 * descriptions, and JSON Schemas defined in tools.ts — demonstrating that the
 * tool contract is transport-independent: the agentic loop in loop.ts
 * consumes it directly, and MCP clients consume it via tools/list +
 * tools/call.
 *
 * Structured errors survive the transport: a ToolError is serialized to the
 * same {ok, errorCategory, isRetryable, message, details} JSON and flagged
 * isError=true in the MCP tool result.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { BankBackend } from "./backend.js";
import { ToolError } from "./errors.js";
import { TOOL_DEFINITIONS, ToolExecutor, type ToolInput } from "./tools.js";

export function buildServer(): Server {
  const backend = new BankBackend();
  const executor = new ToolExecutor(backend);
  const server = new Server(
    { name: "anybank-support", version: "1.0.0" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOL_DEFINITIONS.map((t) => ({
      name: t.name,
      description: t.description,
      inputSchema: t.input_schema as { type: "object"; [key: string]: unknown },
    })),
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    try {
      const output = executor.execute(name, (args ?? {}) as ToolInput);
      return { content: [{ type: "text", text: output }] };
    } catch (err) {
      if (err instanceof ToolError) {
        // MCP signals failure via isError on the result; the structured
        // payload rides in the content, same as the direct-API path.
        return { content: [{ type: "text", text: err.toJson() }], isError: true };
      }
      throw err;
    }
  });

  return server;
}

async function main(): Promise<void> {
  const server = buildServer();
  await server.connect(new StdioServerTransport());
  console.error("anybank-support MCP server running on stdio");
}

if (process.argv[1]?.endsWith("mcpServer.ts")) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
