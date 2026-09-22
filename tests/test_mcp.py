import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.anyio
async def test_mcp_search():
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.mcp_server"],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]

            assert "search" in tool_names

            result = await session.call_tool(
                "search",
                {
                    "query": "What embedding model does Universal RAG use?",
                    "limit": 3,
                },
            )

            assert result.isError is False
            assert result.content
