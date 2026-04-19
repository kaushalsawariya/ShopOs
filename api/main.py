"""
api/main.py
FastAPI backend for ShopOS with auth, memory-aware chat, and MCP endpoints.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.graph import run_agent
from core.tracing import setup_langsmith
from db.database import SessionLocal, init_db
from db.seed import seed_database
from mcp.tools import (
    analyze_bill_file,
    check_overdue_invoices,
    get_customer_summary,
    get_low_stock_alerts,
    get_sales_summary,
    query_database,
    search_policies,
)
from rag.rag_pipeline import index_documents, is_indexed
from services.auth import authenticate_user, create_session, create_user, get_user_by_token, revoke_session
from services.memory import build_memory_context


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_langsmith()
    init_db()
    seed_database()
    if not is_indexed():
        index_documents()
    yield


app = FastAPI(
    title="ShopOS API",
    description="FastAPI backend with authentication, memory, LangGraph agents, and MCP tools.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SignupRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=120)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=120)
    password: str = Field(min_length=6, max_length=128)


class AuthResponse(BaseModel):
    token: str
    user: dict[str, Any]


class ChatRequest(BaseModel):
    query: str
    context: dict[str, Any] | None = None
    file_base64: str | None = None
    filename: str | None = None


class ChatResponse(BaseModel):
    response: str
    route: str | None = None
    tools_used: list[str] = []
    plan: str | None = None
    reflection: str | None = None


class SQLQueryRequest(BaseModel):
    question: str


class PolicySearchRequest(BaseModel):
    query: str


class BillAnalysisRequest(BaseModel):
    file_base64: str
    filename: str = "bill.jpg"


class CustomerSummaryRequest(BaseModel):
    customer_name: str


def serialize_user(user) -> dict[str, Any]:
    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "preferred_agent": user.preferred_agent,
    }


def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    with SessionLocal() as db:
        user = get_user_by_token(db, token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired session.")
        return {"user": user, "token": token}


def run_tool(tool_callable, *args):
    return asyncio.to_thread(tool_callable, *args)


@app.get("/")
async def root():
    return {
        "message": "Welcome to ShopOS API",
        "version": "2.0.0",
        "docs": "/docs",
        "auth": ["/auth/signup", "/auth/login", "/auth/me"],
        "chat": "/chat",
        "tools": "/mcp/tools-list",
    }


@app.post("/auth/signup", response_model=AuthResponse)
async def signup(request: SignupRequest):
    with SessionLocal() as db:
        try:
            user = create_user(db, request.full_name, request.email, request.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session = create_session(db, user)
        return AuthResponse(token=session.token, user=serialize_user(user))


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    with SessionLocal() as db:
        user = authenticate_user(db, request.email, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        session = create_session(db, user)
        return AuthResponse(token=session.token, user=serialize_user(user))


@app.post("/auth/logout")
async def logout(current=Depends(get_current_user)):
    with SessionLocal() as db:
        revoke_session(db, current["token"])
    return {"ok": True}


@app.get("/auth/me")
async def me(current=Depends(get_current_user)):
    with SessionLocal() as db:
        memory = build_memory_context(db, current["user"].id, current["token"])
    return {"user": serialize_user(current["user"]), "memory": memory}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current=Depends(get_current_user)):
    history = (request.context or {}).get("history", [])
    result = await asyncio.to_thread(
        run_agent,
        request.query,
        history,
        request.file_base64,
        request.filename,
        current["user"].id,
        current["token"],
    )
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return ChatResponse(**result)


@app.post("/mcp/sql-query")
async def sql_query_endpoint(request: SQLQueryRequest, _=Depends(get_current_user)):
    return {"result": await run_tool(query_database, request.question), "tool": "query_database"}


@app.post("/mcp/policy-search")
async def policy_search_endpoint(request: PolicySearchRequest, _=Depends(get_current_user)):
    return {"result": await run_tool(search_policies, request.query), "tool": "search_policies"}


@app.post("/mcp/analyze-bill")
async def analyze_bill_endpoint(request: BillAnalysisRequest, _=Depends(get_current_user)):
    return {
        "result": await run_tool(analyze_bill_file, request.file_base64, request.filename),
        "tool": "analyze_bill_file",
    }


@app.post("/mcp/customer-summary")
async def customer_summary_endpoint(request: CustomerSummaryRequest, _=Depends(get_current_user)):
    return {
        "result": await run_tool(get_customer_summary, request.customer_name),
        "tool": "get_customer_summary",
    }


@app.get("/mcp/sales-summary")
async def sales_summary_endpoint(_=Depends(get_current_user)):
    return {"result": await run_tool(get_sales_summary), "tool": "get_sales_summary"}


@app.get("/mcp/overdue-invoices")
async def overdue_invoices_endpoint(_=Depends(get_current_user)):
    return {"result": await run_tool(check_overdue_invoices), "tool": "check_overdue_invoices"}


@app.get("/mcp/low-stock-alerts")
async def low_stock_alerts_endpoint(_=Depends(get_current_user)):
    return {"result": await run_tool(get_low_stock_alerts), "tool": "get_low_stock_alerts"}


@app.get("/mcp/tools-list")
async def list_mcp_tools():
    return {
        "tools": [
            {"name": "query_database", "endpoint": "/mcp/sql-query", "method": "POST"},
            {"name": "search_policies", "endpoint": "/mcp/policy-search", "method": "POST"},
            {"name": "analyze_bill_file", "endpoint": "/mcp/analyze-bill", "method": "POST"},
            {"name": "get_customer_summary", "endpoint": "/mcp/customer-summary", "method": "POST"},
            {"name": "get_sales_summary", "endpoint": "/mcp/sales-summary", "method": "GET"},
            {"name": "check_overdue_invoices", "endpoint": "/mcp/overdue-invoices", "method": "GET"},
            {"name": "get_low_stock_alerts", "endpoint": "/mcp/low-stock-alerts", "method": "GET"},
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
