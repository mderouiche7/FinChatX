import asyncio
import os
from fastmcp.client import Client
from fastmcp.client.transports import PythonStdioTransport
from pprint import pprint



# Define example inputs for each tool
tool_inputs = {
    "get_stock_price": {"ticker": "AAPL"},
    "get_market_movers": {"limit_per_category": 3},
    "get_news": {"ticker": "AAPL"},
}


async def test_connection_and_tools():
    server_script_path = os.path.abspath("qc_server.py") #path to mcp server 

    try:
        async with Client(transport=PythonStdioTransport(script_path=server_script_path)) as client:
            print("**Connected successfully to MCP server over stdio!**\n")

            tools = await client.list_tools()
            print(f"--Discovered {len(tools)} tool(s) on server:--")
            if tools:
                for tool in tools:
                    print(f"- {tool.name}: {tool.description}")
            else:
                print(" No tools found on the server!!!")


             # --- Test Health Check FIRST ---
            print("\n=== Testing Health Check ===")
            try:
                # List available resources to see what's actually available
                available_resources = await client.list_resources()
                print("Available resources:", [r.name for r in available_resources])
                
                # Try to read the health resource - check what URI format is expected
                health_result = await client.read_resource("health://check")
                print("Health check success:", health_result)
            except Exception as e:
                print(f"Health resource error: {type(e).__name__}: {e}")
                
            
            # --- Call each tool safely ---
            if tools:
                print("\n--- Testing tools ---")
                for tool in tools:
                    try:
                        example_input = tool_inputs.get(tool.name, {})

                        print(f"\nCalling tool {tool.name} with input: {example_input}")
                        output = await client.call_tool(tool.name, example_input)

                        # Extract data safely from CallToolResult
                        data = getattr(output, "result", getattr(output, "data", {}))

                        # Limit news output to 10 articles
                        if tool.name == "get_news" and "articles" in data:
                            data["articles"] = data["articles"][:10]

                        print("Tool output (truncated if necessary):")
                        pprint(data)
                    except Exception as e:
                        print(f" Error calling {tool.name}: {type(e).__name__} - {e}")
            else:
                print("\n No tools available to test.") 
    except Exception as conn_err:
        print(f" Connection failed: {conn_err}")

    print("\n Client session closed.")

if __name__ == "__main__":
    asyncio.run(test_connection_and_tools())
