
import os, asyncio
import json
import logging
import traceback
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("fin_langchain_groq")

# Validate GROQ key early (caller can still import but init will check)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# FastMCP client
from fastmcp import Client
from fastmcp.exceptions import ClientError
from mcp.types import TextContent

# LangChain + Groq
try:
    from langchain_groq import ChatGroq
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import SystemMessage
    from langchain.tools import StructuredTool
    from langchain import hub
    from langchain_core.exceptions import OutputParserException
except Exception as e:
    # Import errors should be handled by the caller environment (e.g., Streamlit will show them)
    log.error(f"LangChain/Groq import failed: {e}")
    raise

# Pydantic
from pydantic import BaseModel, Field

# Defaults / Config
DEFAULT_MCP_TARGET = "fin_server_v2_groq.py"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_MEMORY_K = 5

# --- MCP helpers ---
async def call_mcp_tool(mcp_target: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Asynchronously call a FastMCP tool and return parsed content or raise."""
    log.info(f"MCP call -> {tool_name} args={arguments} target={mcp_target}")
    try:
        async with Client(mcp_target) as client:
            result = await client.call_tool(tool_name, arguments, _return_raw_result=True)
            if result.isError:
                msg = None
                if result.content and isinstance(result.content[0], TextContent):
                    msg = result.content[0].text
                log.error(f"MCP tool error {tool_name}: {msg}")
                raise Exception(f"MCP tool error: {msg}")
            if not result.content:
                return {"mcp_result": "no content"}
            first = result.content[0]
            if isinstance(first, TextContent):
                try:
                    return json.loads(first.text)
                except json.JSONDecodeError:
                    return {"mcp_result": first.text}
            else:
                return {"mcp_result": f"non-text content: {type(first)}"}
    except (ClientError, ConnectionRefusedError, FileNotFoundError) as e:
        log.error(f"MCP connection error calling {tool_name}: {e}")
        raise

async def read_mcp_resource(mcp_target: str, uri: str) -> Any:
    log.info(f"MCP read -> {uri} target={mcp_target}")
    try:
        async with Client(mcp_target) as client:
            result = await client.read_resource(uri)
            if not result:
                return {"mcp_result": "resource empty"}
            first = result[0]
            if hasattr(first, "text") and isinstance(first.text, str):
                try:
                    return json.loads(first.text)
                except json.JSONDecodeError:
                    return {"mcp_result": first.text}
            elif hasattr(first, "blob"):
                return {"mcp_result": f"binary len={len(first.blob)}"}
            else:
                return {"mcp_result": f"unknown content {type(first)}"}
    except (ClientError, ConnectionRefusedError, FileNotFoundError) as e:
        log.error(f"MCP read error {uri}: {e}")
        raise

# --- Pydantic schemas for StructuredTool args ---
class GetPriceSchema(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (e.g., AAPL)")

class GetNewsSchema(BaseModel):
    ticker: Optional[str] = Field(None, description="Optional ticker; omit for general market news")

class GetMarketMoversSchema(BaseModel):
    limit_per_category: int = Field(5, description="Max returned items per category")

# --- Tool logic coroutines (they capture mcp_target at runtime via closures) ---
def make_tool_coroutines(mcp_target: str):
    async def _get_price_logic(ticker: str) -> Any:
        """Fetch current stock price for a ticker via MCP."""
        return await call_mcp_tool(mcp_target, "get_stock_price", {"ticker": ticker})

    #News Logic
    async def _get_news_logic(ticker: Optional[str] = None) -> Any:
            """
            Calls the MCP tool get_ticker_news_tool if ticker is provided.
            Returns error if no ticker (general news not yet implemented).
            """
            if ticker:
                ticker = ticker.upper()
                return await call_mcp_tool(mcp_target, "get_ticker_news_tool", {"ticker": ticker})
            else:
                return {"error": "General market news not implemented yet; please provide a ticker."}

    async def _get_market_movers_logic(limit_per_category: int = 5) -> Any:
        return await call_mcp_tool(mcp_target, "get_market_movers", {"limit_per_category": limit_per_category})

    return _get_price_logic, _get_news_logic, _get_market_movers_logic

# --- Factory to build StructuredTool list given an mcp target ---
# --- Factory to build StructuredTool list given an mcp target ---
def build_tools_list(mcp_target: str):
    """
    Builds a list of LangChain StructuredTool objects using coroutines
    that call the MCP backend for price, news, and market movers.
    """
    # Get the coroutines from make_tool_coroutines
    _get_price_logic, _get_news_logic, _get_market_movers_logic = make_tool_coroutines(mcp_target)

    tools = [
        StructuredTool.from_function(
            func=None,
            coroutine=_get_price_logic,
            name="get_price",
            description="Get current stock price for a ticker symbol (e.g., AAPL).",
            args_schema=GetPriceSchema,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=None,
            coroutine=_get_news_logic,
            name="get_news",
            description=(
                "Get recent news for a ticker. Provide ticker symbol to fetch company-specific news. "
                "General market news not implemented yet."
            ),
            args_schema=GetNewsSchema,
            return_direct=False,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=None,
            coroutine=_get_market_movers_logic,
            name="get_market_movers",
            description="Get top gainers, top losers, and most actively traded stocks.",
            args_schema=GetMarketMoversSchema,
            return_direct=False,
            handle_tool_error=True,
        ),
    ]

    return tools



# --- Initialize AgentExecutor (call this from Streamlit UI) ---

def initialize_agent_executor(
    mcp_target: str = DEFAULT_MCP_TARGET,
    groq_model: str = DEFAULT_GROQ_MODEL,
    memory_k: int = DEFAULT_MEMORY_K,
) -> AgentExecutor:
    """Synchronous initializer for AgentExecutor. This function will create the LLM, tools, prompt template,
    and return a configured AgentExecutor instance. Caller may store it in Streamlit session state.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not found in environment. Set it before initializing the agent.")

    log.info(f"Initializing AgentExecutor with model={groq_model} mcp_target={mcp_target}")

    # Build tools
    tools = build_tools_list(mcp_target)

    # LLM: ChatGroq
    llm = ChatGroq(model=groq_model, temperature=0.1)

    # Prompt template: try to pull from hub, else simple fallback
    try:
        prompt_template = hub.pull("hwchase17/openai-tools-agent")
    except Exception:
        log.warning("Could not pull agent prompt from hub; using inline fallback.")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    system_message_content = (
        "You are a helpful and conversational financial data assistant.\n"
        "Available tools: get_price(ticker), get_news(ticker=None), get_market_movers(limit_per_category).\n"
        "Do not provide investment advice. If a tool should be used, call it with correct arguments and then synthesize results."
    )

    # Ensure system message inserted
    if prompt_template.messages and isinstance(prompt_template.messages[0], SystemMessage):
        prompt_template.messages[0].content = system_message_content
    else:
        prompt_template.messages.insert(0, SystemMessage(content=system_message_content))

    # Ensure placeholders
    if "chat_history" not in getattr(prompt_template, "input_variables", []):
        prompt_template.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
    if "agent_scratchpad" not in getattr(prompt_template, "input_variables", []):
        prompt_template.messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))

    memory = ConversationBufferWindowMemory(k=memory_k, memory_key="chat_history", return_messages=True, output_key="output")

    agent = create_openai_tools_agent(llm, tools, prompt_template)

    executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True, max_iterations=5)

    return executor

# --- Suggestion generation using Groq ---
async def generate_suggestions(history: List[Dict[str, str]], groq_model: str = DEFAULT_GROQ_MODEL) -> List[str]:
    """Given a short history (list of dicts with 'role' and 'content'), return up to 3 suggested follow-ups."""
    if not history or len(history) < 2:
        return []
    last_user = next((m for m in reversed(history) if m.get("role") == "user"), None)
    last_assist = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
    if not last_user or not last_assist:
        return []

    llm = ChatGroq(model=groq_model, temperature=0.6)
    system = {"role": "system", "content": "Suggest exactly 3 follow-up user questions as a JSON array of strings; return only the JSON array."}
    human = {"role": "user", "content": f"User: {last_user['content']}\nAssistant: {last_assist['content']}"}
    try:
        resp = await llm.ainvoke([system, human])
        content = resp.content
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed[:3]
        except json.JSONDecodeError:
            log.warning("Suggestions: LLM returned non-JSON content")
            return []
    except Exception as e:
        log.error(f"Error generating suggestions: {e}")
    return []

# --- Simple formatter ---

def format_display_response(text: str) -> str:
    return text


# === end of fin_langchain_groq.py ===

