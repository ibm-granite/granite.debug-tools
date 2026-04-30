"""MCP resources for the perfbench server.

Register resources using the `@mcp.resource()` decorator. Resources expose data
to LLMs — similar to GET endpoints in a REST API. They provide data but should
not perform significant computation or have side effects.
"""

from perfbench.server import mcp


@mcp.resource("info://server")
def server_info() -> str:
    """Return basic information about this MCP server."""
    return "perfbench v0.1.0 — MCP server for Granite benchmarking."
