import os
import httpx
import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# --- FastMCP ---
from fastmcp import FastMCP, Context

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Load Configuration ---
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    finnhub_api_key: Optional[str] = Field(None, validation_alias="FINNHUB_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(None, validation_alias="ALPHA_VANTAGE_API_KEY")

settings = Settings()

if not settings.finnhub_api_key:
    log.warning("FINNHUB_API_KEY not found. News features will be limited.")
if not settings.alpha_vantage_api_key:
    log.warning("ALPHA_VANTAGE_API_KEY not found. Price/movers features will not work.")

# --- FastMCP Server ---
mcp = FastMCP(
    name="FinanceDataServerV2_Groq",
    instructions="Provides stock prices, market news, and top market movers for Groq + LangChain agent."
)

# --- Models ---
class StockQuote(BaseModel):
    ticker: str
    price: float
    change: float
    percent_change: float
    day_high: float
    day_low: float
    day_open: float
    previous_close: float
    source: str
    timestamp: Optional[int] = None

class NewsArticle(BaseModel):
    category: str
    datetime: int
    headline: str
    id: int
    image: str
    related: str
    source: str
    summary: str
    url: str

class MarketMover(BaseModel):
    ticker: str
    price: str
    change_amount: str
    change_percentage: str
    volume: str

class MarketMoversData(BaseModel):
    metadata: Optional[str] = None
    last_updated: Optional[str] = None
    top_gainers: List[MarketMover] = Field(default_factory=list)
    top_losers: List[MarketMover] = Field(default_factory=list)
    most_actively_traded: List[MarketMover] = Field(default_factory=list)

# --- HTTP Client ---
http_client = httpx.AsyncClient(timeout=15.0)

# --- Tools ---
# Stock price tool
@mcp.tool()
async def get_stock_price(ticker: str, ctx: Context) -> Dict[str, Any]:
    ticker = ticker.upper()
    await ctx.info(f"Tool Call: get_stock_price for {ticker}")

    # Finnhub first
    if settings.finnhub_api_key:
        try:
            r = await http_client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": ticker, "token": settings.finnhub_api_key},
            )
            r.raise_for_status()
            data = r.json()
            if data.get("c"):
                quote = StockQuote(
                    ticker=ticker,
                    price=data.get("c", 0.0),
                    change=data.get("d", 0.0),
                    percent_change=data.get("dp", 0.0),
                    day_high=data.get("h", 0.0),
                    day_low=data.get("l", 0.0),
                    day_open=data.get("o", 0.0),
                    previous_close=data.get("pc", 0.0),
                    source="Finnhub",
                    timestamp=data.get("t"),
                )
                await ctx.info(f"Success from Finnhub for {ticker}")
                return quote.model_dump()
        except Exception as e:
            await ctx.warning(f"Finnhub fetch failed for {ticker}: {e}")

    # Alpha Vantage fallback
    if settings.alpha_vantage_api_key:
        try:
            params = {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": settings.alpha_vantage_api_key}
            r = await http_client.get("https://www.alphavantage.co/query", params=params)
            r.raise_for_status()
            data = r.json().get("Global Quote", {})
            if data.get("05. price"):
                quote = StockQuote(
                    ticker=ticker,
                    price=float(data.get("05. price", 0.0)),
                    change=float(data.get("09. change", 0.0)),
                    percent_change=float(data.get("10. change percent", "0%").rstrip("%")),
                    day_high=float(data.get("03. high", 0.0)),
                    day_low=float(data.get("04. low", 0.0)),
                    day_open=float(data.get("02. open", 0.0)),
                    previous_close=float(data.get("08. previous close", 0.0)),
                    source="Alpha Vantage",
                )
                await ctx.info(f"Success from Alpha Vantage for {ticker}")
                return quote.model_dump()
        except Exception as e:
            await ctx.warning(f"Alpha Vantage fetch failed for {ticker}: {e}")

    await ctx.error(f"Failed to fetch stock price for {ticker}")
    return {"error": f"Could not fetch stock price for {ticker}"}

#Ticker news tool
@mcp.tool()
async def get_ticker_news_tool(ticker: str, ctx: Context) -> List[Dict[str, Any]]:
    ticker = ticker.upper()
    if not settings.finnhub_api_key:
        await ctx.error("Finnhub API key missing.")
        return [{"error": "Finnhub API key missing"}]
    try:
        today = date.today(); one_week_ago = today - timedelta(days=7)
        r = await http_client.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": one_week_ago.isoformat(), "to": today.isoformat(), "token": settings.finnhub_api_key}
        )
        r.raise_for_status()
        news_list = r.json()[:10]
        validated = []
        for n in news_list:
            try: validated.append(NewsArticle(**n).model_dump())
            except: continue
        return validated
    except Exception as e: await ctx.warning(f"Error fetching news for {ticker}: {e}"); return [{"error": str(e)}]


# Market movers tool
@mcp.tool()
async def get_market_movers(ctx: Context, limit_per_category: int = 5) -> Dict[str, Any]:
    await ctx.info(f"Tool Call: get_market_movers (limit: {limit_per_category})")
    if not settings.alpha_vantage_api_key:
        await ctx.error("Alpha Vantage API key not configured")
        return {"error": "Alpha Vantage API key not configured"}
    try:
        params = {"function": "TOP_GAINERS_LOSERS", "apikey": settings.alpha_vantage_api_key}
        r = await http_client.get("https://www.alphavantage.co/query", params=params)
        r.raise_for_status()
        data = r.json()
        movers = MarketMoversData.model_validate(data)
        output_data = {
            "top_gainers": [g.model_dump() for g in movers.top_gainers[:limit_per_category]],
            "top_losers": [l.model_dump() for l in movers.top_losers[:limit_per_category]],
            "most_actively_traded": [a.model_dump() for a in movers.most_actively_traded[:limit_per_category]],
            "metadata": movers.metadata,
            "last_updated": movers.last_updated,
        }
        await ctx.info("Market movers fetched successfully")
        return output_data
    except Exception as e:
        await ctx.error(f"Failed fetching market movers: {e}")
        return {"error": "Failed fetching market movers"}

# --- Main ---
if __name__ == "__main__":
    async def close_http():
        if not http_client.is_closed:
            await http_client.aclose()
            log.info("HTTP client closed")

    async def main_run():
        try:
            await mcp.run_async()
        finally:
            await close_http()

    try:
        asyncio.run(main_run())
    except KeyboardInterrupt:
        log.info("Server stopped by user")
