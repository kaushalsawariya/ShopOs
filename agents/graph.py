"""
agents/graph.py
LangGraph-based ShopOS assistant with planning and reflection hooks.
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from db.database import SessionLocal
from guardrails.guardrails import run_guardrails, sanitize_output
from mcp.tools import (
    ALL_TOOLS,
    analyze_bill_file,
    check_overdue_invoices,
    get_customer_summary,
    get_low_stock_alerts,
    get_sales_summary,
    query_database,
    search_policies,
)
from services.memory import build_memory_context, extract_reflection_note, remember_turn, upsert_long_term_memory

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
VALID_AGENTS = {"sql_agent", "rag_agent", "bill_agent", "analytics_agent", "general_agent"}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    next_agent: str
    current_agent: str
    final_answer: str
    file_base64: str | None
    file_name: str | None
    user_id: int | None
    session_token: str | None
    memory_context: dict
    plan: str
    reflection: str


def make_llm(tools: list | None = None):
    llm = ChatOpenAI(model=MODEL, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    return llm.bind_tools(tools) if tools else llm


def format_memory_context(memory_context: dict) -> str:
    short_term = memory_context.get("short_term", [])
    long_term = memory_context.get("long_term", [])
    short_block = "\n".join(
        f"- {item['role']}: {item['content'][:160]}"
        for item in short_term
    ) or "- No recent session memory."
    long_block = "\n".join(
        f"- {item['category']}: {item['summary'][:160]}"
        for item in long_term
    ) or "- No durable user memory."
    return f"Recent memory:\n{short_block}\n\nLong-term memory:\n{long_block}"


def infer_preferred_agent(user_text: str, has_file: bool) -> str:
    if has_file:
        return "bill_agent"
    text = user_text.lower()
    if any(word in text for word in ["policy", "return", "warranty", "payment terms"]):
        return "rag_agent"
    if any(word in text for word in ["summary", "dashboard", "overdue", "stock", "inventory"]):
        return "analytics_agent"
    if any(word in text for word in ["customer", "sale", "sales", "invoice", "product", "balance"]):
        return "sql_agent"
    return "general_agent"


def build_plan(state: AgentState) -> str:
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    last_text = str(last_human.content) if last_human else ""
    preferred = infer_preferred_agent(last_text, bool(state.get("file_base64")))
    steps = [
        "1. Read the latest request and any saved memory.",
        f"2. Start with {preferred} unless the request clearly needs another specialist.",
        "3. Use tools only when they add real shop data or document evidence.",
        "4. Keep the answer concise and operationally useful.",
    ]
    return "\n".join(steps)


def planner_node(state: AgentState) -> AgentState:
    return {**state, "plan": build_plan(state)}


SUPERVISOR_SYSTEM = """Route user requests to the right agent.

- sql_agent: sales, customers, invoices, products, expenses, stock
- rag_agent: policies, returns, warranties, credit terms, payments
- bill_agent: uploaded bill/invoice files (images/PDFs)
- analytics_agent: summaries, overdue invoices, low stock, dashboard
- general_agent: greetings, clarifications, multi-part questions

Consider the plan and memory context.
Respond with ONLY the agent name."""


def supervisor_node(state: AgentState) -> AgentState:
    if state.get("file_base64"):
        return {**state, "next_agent": "bill_agent"}

    llm = make_llm()
    last_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_msg:
        return {**state, "next_agent": "general_agent"}

    prompt = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        SystemMessage(content=f"Plan:\n{state.get('plan', '')}"),
        SystemMessage(content=format_memory_context(state.get("memory_context", {}))),
        HumanMessage(content=str(last_msg.content)),
    ]
    resp = llm.invoke(prompt)
    agent = resp.content.strip().lower()
    return {**state, "next_agent": agent if agent in VALID_AGENTS else "general_agent"}


def supervisor_router(state: AgentState) -> str:
    return state["next_agent"]


def compose_system_prompt(base_prompt: str, state: AgentState) -> list[SystemMessage]:
    return [
        SystemMessage(content=base_prompt),
        SystemMessage(content=f"Execution plan:\n{state.get('plan', '')}"),
        SystemMessage(content=format_memory_context(state.get("memory_context", {}))),
    ]


def invoke_agent(state: AgentState, system_prompt: str, tools: list, agent_name: str) -> AgentState:
    llm = make_llm(tools)
    response = llm.invoke(compose_system_prompt(system_prompt, state) + state["messages"])
    return {**state, "messages": [response], "current_agent": agent_name}


def sql_agent_node(state: AgentState) -> AgentState:
    return invoke_agent(
        state,
        "Use query_database or get_customer_summary for real data. Format currency as Rs. Be concise.",
        [query_database, get_customer_summary],
        "sql_agent",
    )


def rag_agent_node(state: AgentState) -> AgentState:
    return invoke_agent(
        state,
        "Use search_policies for return policies, warranties, credit terms, and payment policies.",
        [search_policies],
        "rag_agent",
    )


def bill_agent_node(state: AgentState) -> AgentState:
    file_base64 = state.get("file_base64")
    if file_base64:
        from core.bill_analyzer import BillAnalyzer

        analyzer = BillAnalyzer()
        response_text = analyzer.analyze_bill(file_base64, state.get("file_name") or "uploaded_bill")
        return {
            **state,
            "messages": [AIMessage(content=response_text)],
            "current_agent": "bill_agent",
        }

    return invoke_agent(
        state,
        (
            "You are an expert at analysing bills and invoices. "
            "Use analyze_bill_file when a file is provided and present extracted information clearly."
        ),
        [analyze_bill_file],
        "bill_agent",
    )


def analytics_agent_node(state: AgentState) -> AgentState:
    return invoke_agent(
        state,
        (
            "You are a shop analytics specialist. Use get_sales_summary, "
            "check_overdue_invoices, and get_low_stock_alerts for real data."
        ),
        [get_sales_summary, check_overdue_invoices, get_low_stock_alerts],
        "analytics_agent",
    )


def general_agent_node(state: AgentState) -> AgentState:
    return invoke_agent(
        state,
        "You are a helpful AI assistant for a shop accounts management system.",
        ALL_TOOLS,
        "general_agent",
    )


tool_node = ToolNode(ALL_TOOLS)


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "reflect"


def route_back_to_current_agent(state: AgentState) -> str:
    current_agent = state.get("current_agent", "general_agent")
    return current_agent if current_agent in VALID_AGENTS else "general_agent"


def reflect_node(state: AgentState) -> AgentState:
    final_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content),
        None,
    )
    reflection = "No durable insight extracted."
    if final_msg and state.get("user_id") and state.get("session_token"):
        last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        insight = extract_reflection_note(str(last_human.content), str(final_msg.content)) if last_human else None
        if insight:
            with SessionLocal() as db:
                upsert_long_term_memory(db, state["user_id"], insight)
            reflection = insight["summary"]
    return {**state, "reflection": reflection, "final_answer": str(final_msg.content) if final_msg else ""}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("bill_agent", bill_agent_node)
    graph.add_node("analytics_agent", analytics_agent_node)
    graph.add_node("general_agent", general_agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("reflect", reflect_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "bill_agent": "bill_agent",
            "analytics_agent": "analytics_agent",
            "general_agent": "general_agent",
        },
    )

    for agent in VALID_AGENTS:
        graph.add_conditional_edges(agent, should_continue, {"tools": "tools", "reflect": "reflect"})

    graph.add_conditional_edges(
        "tools",
        route_back_to_current_agent,
        {agent: agent for agent in VALID_AGENTS},
    )
    graph.add_edge("reflect", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(
    user_message: str,
    history: list[dict] | None = None,
    file_base64: str | None = None,
    file_name: str | None = None,
    user_id: int | None = None,
    session_token: str | None = None,
) -> dict:
    passed, reason = run_guardrails(user_message)
    if not passed:
        return {"response": f"Request blocked: {reason}", "route": "guardrail", "tools_used": [], "error": reason}

    messages: list[BaseMessage] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    if file_base64:
        content = [{"type": "text", "text": f"{user_message}\n\n[FILE_ATTACHED]: Document uploaded for analysis"}]
        messages.append(HumanMessage(content=content))
    else:
        messages.append(HumanMessage(content=user_message))

    with SessionLocal() as db:
        memory_context = build_memory_context(db, user_id, session_token)

    initial_state: AgentState = {
        "messages": messages,
        "next_agent": "general_agent",
        "current_agent": "general_agent",
        "final_answer": "",
        "file_base64": file_base64,
        "file_name": file_name,
        "user_id": user_id,
        "session_token": session_token,
        "memory_context": memory_context,
        "plan": "",
        "reflection": "",
    }

    try:
        result = get_graph().invoke(initial_state)
        final_text = sanitize_output(result.get("final_answer") or "No response generated.")
        tools_used = [
            tc["name"]
            for message in result["messages"]
            if hasattr(message, "tool_calls")
            for tc in (message.tool_calls or [])
        ]

        if user_id and session_token:
            with SessionLocal() as db:
                remember_turn(
                    db=db,
                    user_id=user_id,
                    session_token=session_token,
                    user_message=user_message,
                    assistant_message=final_text,
                    route=result.get("current_agent"),
                )

        return {
            "response": final_text,
            "route": result.get("current_agent", result.get("next_agent", "unknown")),
            "tools_used": list(dict.fromkeys(tools_used)),
            "plan": result.get("plan", ""),
            "reflection": result.get("reflection", ""),
            "error": None,
        }
    except Exception as exc:
        return {
            "response": f"Agent error: {exc}",
            "route": "error",
            "tools_used": [],
            "plan": "",
            "reflection": "",
            "error": str(exc),
        }
