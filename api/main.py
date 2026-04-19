"""
api/main.py
-----------
FastAPI backend integrating MCP tools, LangGraph agents, and RAG pipeline.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any
import asyncio
from agents.graph import run_agent
from mcp.tools import (
    query_database, search_policies, analyze_bill_file,
    get_customer_summary, get_sales_summary,
    check_overdue_invoices, get_low_stock_alerts
)

app = FastAPI(
    title="ShopOs API",
    description="FastAPI backend with MCP tools integration for Shop Accounts Management",
    version="1.0.0"
)

# ── CORS middleware for frontend access ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models for validation ────────────────────────────────────────────

# Chat & Agent endpoints
class ChatRequest(BaseModel):
    query: str
    context: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    route: Optional[str] = None
    tools_used: Optional[List[str]] = None

# MCP Tool endpoints
class SQLQueryRequest(BaseModel):
    question: str

class PolicySearchRequest(BaseModel):
    query: str

class BillAnalysisRequest(BaseModel):
    file_base64: str
    filename: str = "bill.jpg"

class CustomerSummaryRequest(BaseModel):
    customer_name: str

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Welcome to ShopOs API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "sql_query": "/mcp/sql-query",
            "policy_search": "/mcp/policy-search",
            "bill_analysis": "/mcp/analyze-bill",
            "customer_summary": "/mcp/customer-summary",
            "sales_summary": "/mcp/sales-summary",
            "overdue_invoices": "/mcp/overdue-invoices",
            "low_stock_alerts": "/mcp/low-stock-alerts",
            "docs": "/docs"
        }
    }

# ════════════════════════════════════════════════════════════════════════════════
# AGENT ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Multi-agent supervisor chat endpoint.
    Routes to SQL, RAG, Bill, Analytics, or General agent based on query.
    """
    try:
        response = await asyncio.to_thread(run_agent, request.query, request.context)
        return ChatResponse(
            response=response.get("response", ""),
            route=response.get("route"),
            tools_used=response.get("tools_used", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════════════════════════
# MCP TOOL ROUTES (Individual tool access)
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/mcp/sql-query")
async def sql_query_endpoint(request: SQLQueryRequest):
    """
    Text-to-SQL MCP tool endpoint.
    Converts natural language questions into SQL queries.
    """
    try:
        result = await asyncio.to_thread(query_database, request.question)
        return {"result": result, "tool": "query_database"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL Query Error: {str(e)}")


@app.post("/mcp/policy-search")
async def policy_search_endpoint(request: PolicySearchRequest):
    """
    RAG-based policy search MCP tool endpoint.
    Searches terms, conditions, return policies, warranties, etc.
    """
    try:
        result = await asyncio.to_thread(search_policies, request.query)
        return {"result": result, "tool": "search_policies"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Policy Search Error: {str(e)}")


@app.post("/mcp/analyze-bill")
async def analyze_bill_endpoint(request: BillAnalysisRequest):
    """
    Bill/Invoice analysis MCP tool endpoint.
    Analyzes images and PDFs using GPT-4o Vision + RAG context.
    Input: base64-encoded file, filename.
    """
    try:
        result = await asyncio.to_thread(
            analyze_bill_file, request.file_base64, request.filename
        )
        return {"result": result, "tool": "analyze_bill_file"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bill Analysis Error: {str(e)}")


@app.post("/mcp/customer-summary")
async def customer_summary_endpoint(request: CustomerSummaryRequest):
    """
    Customer profile MCP tool endpoint.
    Retrieves contact, credit status, purchase history, and balance.
    """
    try:
        result = await asyncio.to_thread(get_customer_summary, request.customer_name)
        return {"result": result, "tool": "get_customer_summary"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Customer Summary Error: {str(e)}")


@app.get("/mcp/sales-summary")
async def sales_summary_endpoint():
    """
    Sales analytics MCP tool endpoint.
    Retrieves aggregated sales data and trends.
    """
    try:
        result = await asyncio.to_thread(get_sales_summary)
        return {"result": result, "tool": "get_sales_summary"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sales Summary Error: {str(e)}")


@app.get("/mcp/overdue-invoices")
async def overdue_invoices_endpoint():
    """
    Overdue invoices MCP tool endpoint.
    Lists all pending and overdue invoices.
    """
    try:
        result = await asyncio.to_thread(check_overdue_invoices)
        return {"result": result, "tool": "check_overdue_invoices"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overdue Invoices Error: {str(e)}")


@app.get("/mcp/low-stock-alerts")
async def low_stock_alerts_endpoint():
    """
    Low stock alerts MCP tool endpoint.
    Lists products below reorder level.
    """
    try:
        result = await asyncio.to_thread(get_low_stock_alerts)
        return {"result": result, "tool": "get_low_stock_alerts"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Low Stock Alerts Error: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════════
# MCP Tools Metadata
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/mcp/tools-list")
async def list_mcp_tools():
    """
    List all available MCP tools and their descriptions.
    """
    return {
        "tools": [
            {
                "name": "query_database",
                "endpoint": "/mcp/sql-query",
                "method": "POST",
                "description": "Text-to-SQL queries on shop database"
            },
            {
                "name": "search_policies",
                "endpoint": "/mcp/policy-search",
                "method": "POST",
                "description": "RAG-based policy and terms search"
            },
            {
                "name": "analyze_bill_file",
                "endpoint": "/mcp/analyze-bill",
                "method": "POST",
                "description": "Bill/Invoice analysis using Vision + RAG"
            },
            {
                "name": "get_customer_summary",
                "endpoint": "/mcp/customer-summary",
                "method": "POST",
                "description": "Full customer profile and history"
            },
            {
                "name": "get_sales_summary",
                "endpoint": "/mcp/sales-summary",
                "method": "GET",
                "description": "Aggregated sales analytics"
            },
            {
                "name": "check_overdue_invoices",
                "endpoint": "/mcp/overdue-invoices",
                "method": "GET",
                "description": "Pending and overdue invoices"
            },
            {
                "name": "get_low_stock_alerts",
                "endpoint": "/mcp/low-stock-alerts",
                "method": "GET",
                "description": "Products below reorder level"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)