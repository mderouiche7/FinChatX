import streamlit as st
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from fin_langchain_groq import (
    initialize_agent_executor,
    generate_suggestions,
    format_display_response,
    DEFAULT_MCP_TARGET,
    DEFAULT_GROQ_MODEL,
)

st.set_page_config(page_title="Finance Assistant (Groq)", layout="wide")
st.title("📈 Conversational Finance Assistant (Groq)")


# Session state defaults
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "agent_executor" not in st.session_state:
    try:
        st.session_state["agent_executor"] = initialize_agent_executor(
            mcp_target=DEFAULT_MCP_TARGET, groq_model=DEFAULT_GROQ_MODEL
        )
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

if "current_suggestions" not in st.session_state:
    st.session_state["current_suggestions"] = []

# Helper for suggestion clicks
def handle_suggestion_click(sugg: str):
    st.session_state["messages"].append({"role": "user", "content": sugg})
    st.session_state["current_suggestions"] = []

# Async runner helper
def run_async(coro):
    """Run async function in Streamlit safely."""
    try:
        return asyncio.get_event_loop().run_until_complete(coro)
    except RuntimeError:
        # No current loop in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

async def process_user_message(user_message: str):
    executor = st.session_state["agent_executor"]
    result = {"output": "No output returned.", "suggestions": []}
    try:
        resp = await executor.ainvoke({"input": user_message})
        result["output"] = resp.get("output", "No output returned by agent.")

        # Generate follow-up suggestions
        history = st.session_state["messages"][-2:] if len(st.session_state["messages"]) >= 2 else st.session_state["messages"]
        result["suggestions"] = await generate_suggestions(history)
    except Exception as e:
        result["output"] = f"Internal error occurred: {e}"
    return result

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(format_display_response(msg["content"]))

# Display suggestion buttons
if st.session_state["current_suggestions"]:
    st.markdown("---")
    cols = st.columns(len(st.session_state["current_suggestions"]))
    for i, s in enumerate(st.session_state["current_suggestions"]):
        with cols[i]:
            st.button(
                s,
                key=f"sugg_{i}",
                on_click=handle_suggestion_click,
                args=(s,),
                use_container_width=True,
            )

# Chat input
if user_input := st.chat_input("Ask about a ticker, stock price, or market movers:"):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["current_suggestions"] = []

    # Process user message asynchronously
    result = run_async(process_user_message(user_input))

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(format_display_response(result["output"]))

    # Save assistant message and suggestions
    st.session_state["messages"].append({"role": "assistant", "content": result["output"]})
    st.session_state["current_suggestions"] = result["suggestions"]
