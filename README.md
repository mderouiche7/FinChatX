# 💹 FinChatX — Conversational Finance Analyst (LangChain + MCP + Groq)

> **FinChatX** is an **agentic finance assistant** powered by **LangChain**, **Groq API**, **FastMCP**, and **Streamlit**.  
> It acts as an intelligent analyst capable of fetching live stock data, market movers, and financial news —  
> combining **MCP tool orchestration**, **asynchronous workflows**, and a **LangGraph-inspired architecture**.

---

##  Overview

**FinChatX** simulates an **autonomous financial analyst** system integrating modern AI and multi-agent design principles.  
It demonstrates how to combine *LLM reasoning*, *tool calling*, and *async MCP workflows* for real-world financial data retrieval.

-  **LangChain AgentExecutor** for reasoning and decision-making  
-  **MCP (Model Context Protocol)** for dynamic tool routing  
-  **Groq LLMs** (Llama 3.1/3.3) for ultra-fast inference  
-  **Streamlit UI** for user interaction and visualization  
-  **Asynchronous backend** for concurrent execution of tools  

---

##  System Architecture

User Query → Streamlit UI
↓
LangChain AgentExecutor (Groq model)
↓
Async MCP Tools via FastMCP Client
├── get_stock_price (Finnhub / AlphaVantage / yfinance fallback)
├── get_ticker_news_tool (Company news via API)
└── get_market_movers (Top gainers / losers / active stocks)
↓
Response → Streamlit display + contextual follow-up suggestions


## 🧩 Core Components

| File | Description |
|------|--------------|
| **`fin_langchain_groq.py`** | Main LangChain agent logic, Groq model initialization, memory, and async executor. |
| **`fin_server_v2_groq.py`** | MCP-compliant FastAPI server exposing finance tools for data retrieval. |
| **`streamlit_app.py`** | Front-end chat interface with async event loop handling. |
| **`test_file.py`** | Example script to test MCP tool invocation and LangChain agent execution. |
| **`.env`** | Contains API keys (excluded from repo; see `.env.example`). |

---



🧩 **Example Streamlit UI:**  
## UI Preview
![FinChatX Dashboard](C:\DevProjs\FinChatX\data\UI_interface_streamlit.png)

💻 **Test File Example:**

```bash
python test_file.py