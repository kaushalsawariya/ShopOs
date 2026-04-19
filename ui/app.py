"""
ui/app.py
Streamlit frontend for ShopOS with auth, memory-aware chat, and admin tools.
"""

from __future__ import annotations

import base64
import os
import sys
from datetime import datetime

import streamlit as st
import streamlit_mermaid as st_mermaid
from dotenv import load_dotenv
from sqlalchemy import text as sa_text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.graph import run_agent
from core.tracing import setup_langsmith
from db.database import SessionLocal, engine, get_table_schema, init_db
from db.seed import seed_database
from rag.rag_pipeline import clear_vector_store, index_documents, is_indexed
from services.auth import authenticate_user, create_session, create_user, revoke_session
from services.memory import build_memory_context

load_dotenv()


@st.cache_resource(show_spinner="Starting ShopOS...")
def startup() -> bool:
    setup_langsmith()
    init_db()
    seed_database()
    if not is_indexed():
        index_documents()
    return True


startup()

st.set_page_config(page_title="ShopOS", page_icon="🏪", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      :root {
        --bg: #f4efe6;
        --panel: rgba(255,255,255,0.86);
        --ink: #1f2a2e;
        --muted: #617076;
        --line: rgba(80,63,43,0.14);
        --accent: #b85c38;
        --accent-2: #2f6c7a;
      }
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(184, 92, 56, 0.10), transparent 28%),
          radial-gradient(circle at top right, rgba(47, 108, 122, 0.12), transparent 22%),
          linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
        color: var(--ink);
      }
      .surface, .hero, [data-testid="stChatMessage"], [data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 18px;
      }
      .hero {
        background: linear-gradient(135deg, #204e57 0%, #2f6c7a 42%, #e9dcc8 42%, #f8f2e8 100%);
        color: white;
      }
      .metric {
        padding: 12px 14px;
        border-radius: 18px;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(80,63,43,0.12);
      }
      .memory-item {
        padding: 10px 12px;
        border-radius: 14px;
        background: rgba(47,108,122,0.08);
        margin-bottom: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def current_user():
    return st.session_state.get("user")


def current_token():
    return st.session_state.get("session_token")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div style="font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:800;">ShopOS Control Room</div>
          <h1 style="margin:12px 0 8px 0;">Run sales questions, policy search, and invoice analysis from one workspace.</h1>
          <p style="max-width:640px;margin:0;">
            Multi-agent routing, FastAPI backend support, login-protected access, and memory-aware conversations
            without changing the core shop workflows you already have.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def auth_card() -> None:
    st.title("ShopOS Sign In")
    st.caption("Create an account or sign in to use the assistant and save memory.")
    login_tab, signup_tab = st.tabs(["Login", "Sign up"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            with SessionLocal() as db:
                user = authenticate_user(db, email, password)
                if not user:
                    st.error("Invalid email or password.")
                else:
                    session = create_session(db, user)
                    st.session_state.user = {
                        "id": user.id,
                        "full_name": user.full_name,
                        "email": user.email,
                    }
                    st.session_state.session_token = session.token
                    st.session_state.chat_history = []
                    st.success("Logged in successfully.")
                    st.rerun()

    with signup_tab:
        with st.form("signup_form", clear_on_submit=False):
            full_name = st.text_input("Full name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            with SessionLocal() as db:
                try:
                    user = create_user(db, full_name, email, password)
                    session = create_session(db, user)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.user = {
                        "id": user.id,
                        "full_name": user.full_name,
                        "email": user.email,
                    }
                    st.session_state.session_token = session.token
                    st.session_state.chat_history = []
                    st.success("Account created successfully.")
                    st.rerun()


def render_memory_sidebar() -> None:
    user = current_user()
    token = current_token()
    if not user or not token:
        return

    with SessionLocal() as db:
        memory = build_memory_context(db, user["id"], token)

    st.sidebar.markdown(f"### {user['full_name']}")
    st.sidebar.caption(user["email"])
    if st.sidebar.button("Logout", use_container_width=True):
        with SessionLocal() as db:
            revoke_session(db, token)
        for key in ["user", "session_token", "chat_history"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### Memory")
    long_term = memory.get("long_term", [])
    short_term = memory.get("short_term", [])
    if long_term:
        st.sidebar.markdown("**Long-term**")
        for item in long_term:
            st.sidebar.markdown(f"- `{item['category']}`: {item['summary']}")
    else:
        st.sidebar.caption("No long-term memory saved yet.")
    st.sidebar.markdown("**Recent context**")
    if short_term:
        for item in short_term[-4:]:
            st.sidebar.markdown(f"- `{item['role']}`: {item['content'][:60]}")
    else:
        st.sidebar.caption("No recent turns yet.")


def render_assistant_page() -> None:
    st.title("AI Shop Assistant")
    render_hero()

    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="metric"><strong>Agents</strong><br>5 specialist routes</div>', unsafe_allow_html=True)
    col2.markdown('<div class="metric"><strong>Patterns</strong><br>Planning + reflection</div>', unsafe_allow_html=True)
    col3.markdown('<div class="metric"><strong>Memory</strong><br>Short-term + long-term</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.container():
        st.markdown('<div class="surface"><strong>Document intake</strong><br>Upload a bill, invoice, or receipt for analysis.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload bill or invoice",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        label_visibility="collapsed",
    )

    suggestions = [
        "Show me sales summary for this month",
        "Who are the top 5 customers by revenue?",
        "Analyze this uploaded bill",
        "What is the return policy for electronics?",
        "Show me all overdue invoices",
        "What products are low on stock?",
    ]
    st.markdown("### Quick prompts")
    suggestion_columns = st.columns(3)
    for index, suggestion in enumerate(suggestions):
        if suggestion_columns[index % 3].button(suggestion, use_container_width=True):
            st.session_state.prefill_chat = suggestion

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("meta"):
                meta = message["meta"]
                st.caption(
                    f"Route: {meta.get('route', 'n/a')} | "
                    f"Tools: {', '.join(meta.get('tools_used', [])) or 'none'}"
                )

    prompt = st.chat_input("Ask about sales, customers, policies, bills, or inventory...")
    user_input = prompt or st.session_state.pop("prefill_chat", "")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    file_base64 = None
    file_name = None
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > 10 * 1024 * 1024:
            st.error("File too large. Maximum supported size is 10MB.")
            return
        file_base64 = base64.b64encode(file_bytes).decode()
        file_name = uploaded_file.name

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(
                user_message=user_input,
                history=st.session_state.chat_history[:-1],
                file_base64=file_base64,
                file_name=file_name,
                user_id=current_user()["id"],
                session_token=current_token(),
            )
        st.markdown(result["response"])
        st.caption(
            f"Route: {result.get('route', 'n/a')} | "
            f"Tools: {', '.join(result.get('tools_used', [])) or 'none'}"
        )
        if result.get("plan"):
            with st.expander("Planner"):
                st.code(result["plan"])
        if result.get("reflection") and result["reflection"] != "No durable insight extracted.":
            with st.expander("Reflection"):
                st.write(result["reflection"])

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["response"],
            "meta": {
                "route": result.get("route"),
                "tools_used": result.get("tools_used", []),
            },
        }
    )

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()


def render_workflow_page() -> None:
    st.title("Workflow Graph")
    st.caption("Planning, routing, tools, and reflection.")
    st_mermaid.st_mermaid(
        """
        flowchart TD
            A["User Request"] --> B["Planner"]
            B --> C["Supervisor"]
            C --> D["SQL Agent"]
            C --> E["RAG Agent"]
            C --> F["Bill Agent"]
            C --> G["Analytics Agent"]
            C --> H["General Agent"]
            D --> I["MCP Tools"]
            E --> I
            F --> I
            G --> I
            H --> I
            I --> J["Reflector"]
            J --> K["Short-term Memory"]
            J --> L["Long-term Memory"]
            J --> M["Final Answer"]
        """,
        height=480,
    )


def render_settings_page() -> None:
    st.title("Settings & Administration")

    left, right = st.columns(2)
    with left:
        st.subheader("Vector Store")
        if is_indexed():
            st.success("Vector index is ready.")
        else:
            st.warning("Vector index has not been built yet.")
        if st.button("Rebuild Vector Index", type="primary"):
            with st.spinner("Re-indexing documents..."):
                count = index_documents()
            st.success(f"Rebuilt vector store with {count} chunks.")
            st.rerun()
        if st.button("Clear Vector Embeddings"):
            st.info(clear_vector_store())
            st.rerun()

    with right:
        st.subheader("Tracing")
        key = os.getenv("LANGCHAIN_API_KEY", "")
        tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
        project = os.getenv("LANGCHAIN_PROJECT", "shop-accounts-capstone")
        st.write(f"Tracing enabled: `{tracing}`")
        st.write(f"Project: `{project}`")
        st.write("LangSmith key configured." if key and not key.startswith("ls__your") else "LangSmith key not configured.")

    st.divider()
    st.subheader("Database")
    with engine.connect() as conn:
        tables = conn.execute(sa_text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")).fetchall()
    st.write(f"Database: `{os.getenv('DATABASE_URL', 'auto-managed sqlite')}`")
    for (table_name,) in tables:
        with engine.connect() as conn:
            count = conn.execute(sa_text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        st.write(f"- `{table_name}`: {count} rows")
    with st.expander("Full schema"):
        st.code(get_table_schema())


if not current_user():
    auth_card()
else:
    with st.sidebar:
        st.markdown("## ShopOS")
        st.caption(f"Online · {datetime.now().strftime('%H:%M:%S')}")
    render_memory_sidebar()
    page = st.sidebar.radio("Navigation", ["AI Assistant", "Workflow Graph", "Settings"])
    if page == "AI Assistant":
        render_assistant_page()
    elif page == "Workflow Graph":
        render_workflow_page()
    else:
        render_settings_page()
