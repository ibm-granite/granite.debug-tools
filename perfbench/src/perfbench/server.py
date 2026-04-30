"""Main MCP server entry point.

Creates the FastMCP instance and registers all tools, resources, and prompts.
"""

import logging
import os

from mcp.server.fastmcp import FastMCP

_log_level = os.environ.get("MCP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)

mcp = FastMCP("perfbench")

# Import modules to register their tools, resources, and prompts with the server.
# Each module uses the shared `mcp` instance to register its handlers.
from perfbench import prompts, resources, tools  # noqa: E402, F401


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
