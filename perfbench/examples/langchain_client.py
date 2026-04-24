"""LangChain MCP client for the granite.debug.perfbench server.

Connects to the MCP server via stdio and exposes its tools to a LangGraph
agent powered by IBM Granite 4.

The client maintains a **single persistent MCP session** so that server-side
state (e.g. running benchmarks) is preserved across queries.

Supported LLM providers (set via ``LLM_PROVIDER`` env var):

* **ollama** (default) — runs locally, no credentials needed.
  Requires Ollama with the model pulled: ``ollama pull granite4:micro``

* **watsonx** — uses IBM watsonx.ai. Requires:
      WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID
"""

import asyncio
import os
import sys

from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_MODEL = "granite4:micro"
WATSONX_MODEL = "ibm/granite-4-h-small"


def _build_llm():
    """Return a chat model based on the LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=OLLAMA_MODEL), OLLAMA_MODEL

    if provider == "watsonx":
        from langchain_ibm import ChatWatsonx

        return ChatWatsonx(model_id=WATSONX_MODEL), WATSONX_MODEL

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama' or 'watsonx'."
    )


async def interactive_loop() -> None:
    """Run an interactive REPL with a persistent MCP server session."""
    llm, model_name = _build_llm()
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    print("granite.debug.perfbench MCP Client")
    print(f"Provider: {provider}  |  Model: {model_name}")
    print("Type 'quit' or 'exit' to stop.\n")

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "perfbench"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            
            print("=" * 60)
            print("Available tools:")
            print("=" * 60)
            for tool in tools:
                print(f"[{tool.name}]")
                print(f"{tool.description}")
                print("-" * 60)
            print()
            
            agent = create_agent(llm, tools)

            while True:
                try:
                    query = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not query:
                    continue
                if query.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                try:
                    response = await agent.ainvoke(
                        {"messages": query}
                    )
                    messages = response.get("messages", [])
                    if messages:
                        print(f"Agent: {messages[-1].content}\n")
                    else:
                        print(f"Agent: {response}\n")
                except Exception as exc:
                    print(f"Error: {exc}\n", file=sys.stderr)


def main() -> None:
    """Entry point."""
    asyncio.run(interactive_loop())


if __name__ == "__main__":
    main()
