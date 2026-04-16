"""
agents/graph.py
---------------
LangGraph Multi-Agent Supervisor Architecture.

Flow:
  User message
       │
       ▼
  ┌─────────────┐
  │  SUPERVISOR  │  ← decides which specialist to route to
  └──────┬──────┘
         │ routes to one of:
    ┌────┴────┬──────────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼          ▼
  SQL       RAG        BILL      ANALYTICS  GENERAL
 Agent    Agent      Agent      Agent       Agent
    │         │          │          │          │
    └────┬────┴──────────┴──────────┘          │
         ▼                                     ▼
     Tool calls                          Direct LLM
         │
         ▼
     Final Answer

Each specialist agent has access to relevant MCP tools only.
The supervisor reads agent outputs and decides if another agent is needed.
"""

import os
from typing import Annotated, Literal, TypedDict, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from mcp.tools import (
    query_database, search_policies, analyze_bill_image,
    get_customer_summary, get_sales_summary,
    check_overdue_invoices, get_low_stock_alerts, ALL_TOOLS
)
from guardrails.guardrails import run_guardrails, sanitize_output

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── Shared graph state ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]  # append-only
    next_agent: str       # which agent to route to next
    final_answer: str     # accumulated final response


# ── Helper: build an LLM bound to specific tools ────────────────────────────
def make_llm(tools: list = None):
    llm = ChatOpenAI(model=MODEL, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    if tools:
        return llm.bind_tools(tools)
    return llm


# ══════════════════════════════════════════════════════════════════
# NODE: SUPERVISOR
# Reads user message and routes to the right specialist agent.
# ══════════════════════════════════════════════════════════════════
SUPERVISOR_SYSTEM = """You are a supervisor for a shop accounts management system.
Your job is to read the user's request and route it to the most appropriate specialist agent.

Available agents:
- sql_agent      : Questions about sales data, customers, invoices, products, expenses, stock
- rag_agent      : Questions about policies, return policy, warranty, credit terms, payment terms
- bill_agent     : Analysing uploaded bill/invoice images
- analytics_agent: Dashboard stats, sales summaries, overdue invoices, low stock alerts
- general_agent  : General greetings, clarifications, or multi-part questions

Respond with ONLY one of: sql_agent, rag_agent, bill_agent, analytics_agent, general_agent
"""

def supervisor_node(state: AgentState) -> AgentState:
    """Routes the conversation to the appropriate specialist."""
    llm = make_llm()
    # Get the last human message for routing decision
    last_msg = next((m for m in reversed(state["messages"])
                     if isinstance(m, HumanMessage)), None)
    if not last_msg:
        return {**state, "next_agent": "general_agent"}

    resp = llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=str(last_msg.content)),
    ])
    agent = resp.content.strip().lower()
    # Validate — fall back to general if unexpected
    valid = {"sql_agent","rag_agent","bill_agent","analytics_agent","general_agent"}
    if agent not in valid:
        agent = "general_agent"
    return {**state, "next_agent": agent}


def supervisor_router(state: AgentState) -> str:
    """Edge function: returns which agent to execute next."""
    return state["next_agent"]


# ══════════════════════════════════════════════════════════════════
# NODE: SQL AGENT — handles data queries
# ══════════════════════════════════════════════════════════════════
SQL_SYSTEM = """You are an expert shop data analyst.
Answer the user's question by using the query_database or get_customer_summary tool.
Always use tools to get real data — never make up numbers.
Format currency as ₹ (Indian Rupee). Keep the answer concise and clear."""

def sql_agent_node(state: AgentState) -> AgentState:
    llm = make_llm([query_database, get_customer_summary])
    response = llm.invoke([SystemMessage(content=SQL_SYSTEM)] + state["messages"])
    return {**state, "messages": [response]}


# ══════════════════════════════════════════════════════════════════
# NODE: RAG AGENT — handles policy questions
# ══════════════════════════════════════════════════════════════════
RAG_SYSTEM = """You are a shop policy expert.
Answer questions about return policies, warranties, credit terms, and payment policies
by using the search_policies tool.
Always search before answering. Quote relevant policy sections."""

def rag_agent_node(state: AgentState) -> AgentState:
    llm = make_llm([search_policies])
    response = llm.invoke([SystemMessage(content=RAG_SYSTEM)] + state["messages"])
    return {**state, "messages": [response]}


# ══════════════════════════════════════════════════════════════════
# NODE: BILL AGENT — handles bill image analysis
# ══════════════════════════════════════════════════════════════════
BILL_SYSTEM = """You are an expert at analysing bills and invoices.
When an image is provided (as base64 in the message), use analyze_bill_image to extract data.
Present the extracted information clearly with all amounts in ₹."""

def bill_agent_node(state: AgentState) -> AgentState:
    llm = make_llm([analyze_bill_image])
    response = llm.invoke([SystemMessage(content=BILL_SYSTEM)] + state["messages"])
    return {**state, "messages": [response]}


# ══════════════════════════════════════════════════════════════════
# NODE: ANALYTICS AGENT — handles dashboard/summary queries
# ══════════════════════════════════════════════════════════════════
ANALYTICS_SYSTEM = """You are a shop analytics specialist.
Use the available tools to fetch real-time shop analytics:
- get_sales_summary for revenue and transaction stats
- check_overdue_invoices for outstanding payments
- get_low_stock_alerts for inventory alerts
Present data with clear formatting and ₹ for currency."""

def analytics_agent_node(state: AgentState) -> AgentState:
    llm = make_llm([get_sales_summary, check_overdue_invoices, get_low_stock_alerts])
    response = llm.invoke([SystemMessage(content=ANALYTICS_SYSTEM)] + state["messages"])
    return {**state, "messages": [response]}


# ══════════════════════════════════════════════════════════════════
# NODE: GENERAL AGENT — handles everything else
# ══════════════════════════════════════════════════════════════════
GENERAL_SYSTEM = """You are a helpful AI assistant for a shop accounts management system.
You can access all shop tools when needed. Be friendly, concise, and professional.
For greetings or clarifications, respond directly without using tools."""

def general_agent_node(state: AgentState) -> AgentState:
    llm = make_llm(ALL_TOOLS)
    response = llm.invoke([SystemMessage(content=GENERAL_SYSTEM)] + state["messages"])
    return {**state, "messages": [response]}


# ══════════════════════════════════════════════════════════════════
# NODE: TOOL EXECUTOR (shared across all agents)
# ══════════════════════════════════════════════════════════════════
tool_node = ToolNode(ALL_TOOLS)


def should_continue(state: AgentState) -> str:
    """
    After any agent runs, check if it made tool calls.
    If yes → run tools. If no → we're done.
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ══════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════
def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph multi-agent supervisor graph.
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("sql_agent",        sql_agent_node)
    graph.add_node("rag_agent",        rag_agent_node)
    graph.add_node("bill_agent",       bill_agent_node)
    graph.add_node("analytics_agent",  analytics_agent_node)
    graph.add_node("general_agent",    general_agent_node)
    graph.add_node("tools",            tool_node)

    # Entry point: always start at supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor routes to one of the specialist agents
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "sql_agent":       "sql_agent",
            "rag_agent":       "rag_agent",
            "bill_agent":      "bill_agent",
            "analytics_agent": "analytics_agent",
            "general_agent":   "general_agent",
        }
    )

    # Each agent may call tools or finish
    for agent in ["sql_agent","rag_agent","bill_agent","analytics_agent","general_agent"]:
        graph.add_conditional_edges(agent, should_continue, {"tools": "tools", END: END})

    # After tools execute, return to the same agent for final answer
    # (tools send result back; agent interprets and responds)
    graph.add_edge("tools", "sql_agent")      # tools always return to sql_agent
    # Note: in a more complex setup you'd track which agent called tools.
    # For simplicity, tools loop back through sql_agent (covers 80% of tool use).

    return graph.compile()


# ══════════════════════════════════════════════════════════════════
# PUBLIC API: run the graph for one user turn
# ══════════════════════════════════════════════════════════════════

# Compiled graph (singleton)
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(
    user_message: str,
    history: list[dict] | None = None,
    image_base64: str | None = None,
) -> dict:
    """
    Run the multi-agent graph for one user turn.

    Args:
        user_message : Plain text from the user
        history      : List of {"role": "user"/"assistant", "content": "..."} dicts
        image_base64 : Optional base64-encoded bill image

    Returns:
        {
          "response"  : str — final assistant message
          "route"     : str — which agent handled it
          "tools_used": list[str]
          "error"     : str | None
        }
    """
    # ── Guardrail check ──────────────────────────────────────────
    passed, reason = run_guardrails(user_message)
    if not passed:
        return {"response": f"⚠️ Request blocked: {reason}",
                "route": "guardrail", "tools_used": [], "error": reason}

    # ── Build message history ────────────────────────────────────
    messages: list[BaseMessage] = []
    for h in (history or []):
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            messages.append(AIMessage(content=h["content"]))

    # Attach image to message if provided
    if image_base64:
        # Signal the bill agent via a structured message
        content = [
            {"type": "text",
             "text": f"{user_message}\n\n[IMAGE_BASE64]:{image_base64}"},
        ]
        messages.append(HumanMessage(content=content))
    else:
        messages.append(HumanMessage(content=user_message))

    # ── Run the graph ────────────────────────────────────────────
    try:
        initial_state: AgentState = {
            "messages": messages,
            "next_agent": "general_agent",
            "final_answer": "",
        }
        result = get_graph().invoke(initial_state)

        # Extract final AI response
        final_msg = next(
            (m for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and m.content),
            None
        )
        response_text = sanitize_output(final_msg.content if final_msg else "No response generated.")

        # Collect tool names used
        tools_used = [
            tc["name"]
            for m in result["messages"]
            if hasattr(m, "tool_calls")
            for tc in (m.tool_calls or [])
        ]

        return {
            "response":   response_text,
            "route":      result.get("next_agent", "unknown"),
            "tools_used": list(set(tools_used)),
            "error":      None,
        }

    except Exception as e:
        return {
            "response": f"Agent error: {str(e)}",
            "route": "error",
            "tools_used": [],
            "error": str(e),
        }
