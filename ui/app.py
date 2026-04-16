"""
ui/app.py
---------
Streamlit frontend for Shop Accounts Management System.
Pages: Chat (AI), Text-to-SQL, Bill Scanner, Policy Search (RAG),
       Customers, Sales, Dashboard, Settings (clear embeddings).

Run: streamlit run ui/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64, json
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from agents.graph import run_agent

load_dotenv()

# ── One-time startup ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🚀 Starting up system...")
def startup():
    """Initialise DB, seed data, build RAG index. Cached so it runs once."""
    from core.tracing import setup_langsmith
    from db.database import init_db
    from db.seed import seed_database
    from rag.rag_pipeline import index_documents, is_indexed

    setup_langsmith()
    init_db()
    seed_database()
    if not is_indexed():
        index_documents()
    return True


startup()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopOS — Accounts Manager",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Light theme styling */
  [data-testid="stSidebar"] { background: #f8f9fa; border-right: 1px solid #e9ecef; }
  .stButton>button { border-radius: 8px; font-weight: 600; transition: all .15s; background: #ffffff; border: 1px solid #dee2e6; }
  .stButton>button:hover { border-color: #007bff !important; color: #007bff !important; background: #f8f9fa !important; }
  .metric-card {
    background: #ffffff; border: 1px solid #dee2e6; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  .metric-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .8px;
    color: #6c757d; margin-bottom: 4px; }
  .metric-card .value { font-size: 28px; font-weight: 800; color: #212529; line-height: 1.1; }
  .metric-card .sub { font-size: 11px; color: #6c757d; margin-top: 4px; }
  .agent-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; background: rgba(0,123,255,.12);
    color: #007bff; border: 1px solid rgba(0,123,255,.25); margin-left: 8px;
  }
  .tool-chip {
    display: inline-block; padding: 1px 8px; border-radius: 20px;
    font-size: 10px; background: #f8f9fa; color: #495057;
    border: 1px solid #dee2e6; margin: 1px;
  }
  .source-chip {
    display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-size: 11px; background: rgba(0,123,255,.08); color: #007bff;
    border: 1px solid rgba(0,123,255,.2); margin: 2px;
  }
  .sql-block { background: #f8f9fa; border-radius: 8px; padding: 12px;
    font-family: 'DM Mono', monospace; font-size: 13px; color: #007bff;
    border: 1px solid #dee2e6; white-space: pre-wrap; word-break: break-all; }
  .bill-field { background: #ffffff; border-radius: 6px; padding: 10px 14px; margin: 4px 0;
    border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .bill-field .lbl { font-size: 10px; text-transform: uppercase; color: #6c757d; }
  .bill-field .val { font-size: 14px; color: #212529; font-family: monospace; }
  .warning-box { background: rgba(255,193,7,.08); border: 1px solid rgba(255,193,7,.3);
    border-radius: 8px; padding: 12px; color: #856404; font-size: 13px; }
  hr { border-color: #dee2e6 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏪 ShopOS")
    st.markdown("*AI Assistant with Document Analysis*")
    st.divider()

    # System status indicator
    st.markdown("### 🔄 System Status")
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        st.success("🟢")
    with status_col2:
        st.markdown("**Online & Ready**")
        st.caption("All agents active")

    st.divider()

    # Document analysis capabilities
    st.markdown("### 📄 Document Analysis")
    st.markdown("✅ **Images:** JPG, PNG, WebP")
    st.markdown("✅ **PDFs:** Multi-page support")
    st.markdown("✅ **AI Vision:** GPT-4o powered")
    st.markdown("✅ **Rate Limited:** Smart token management")

    st.divider()

    page = st.radio(
        "Navigation",
        ["🤖 AI Assistant", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**LangGraph Flow:**")
    st.markdown("""
```
User Input
    │
  Supervisor
  ├─ sql_agent      → DB queries
  ├─ rag_agent      → Policies
  ├─ bill_agent     → Images
  ├─ analytics_agent→ Stats
  └─ general_agent  → Chat
         │
      Tool Nodes (MCP)
         │
    Final Answer
```
""")
    st.divider()
    now = datetime.now().strftime("%H:%M:%S")
    st.caption(f"🟢 System online · {now}")


# ═════════════════════════════════════════════════════════════════
# PAGE: AI ASSISTANT
# ═════════════════════════════════════════════════════════════════
if page == "🤖 AI Assistant":
    st.title("🤖 AI Shop Assistant")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-radius: 10px; padding: 15px; margin: 10px 0; color: #1565c0; border: 1px solid #90caf9;">
        <h4 style="margin: 0; color: #1565c0;">🧠 Multi-Agent AI Assistant</h4>
        <p style="margin: 5px 0 0 0; color: #1976d2;">
            Ask questions about sales, customers, policies, bills, or upload documents for analysis.
            Powered by LangGraph with specialized agents for different tasks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Image/PDF upload section with improved styling
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
                border-radius: 15px; padding: 20px; margin: 10px 0;
                border: 2px solid #ffcc02; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h4 style="color: #e65100; margin: 0; text-align: center;">
            📎 Upload Bill/Invoice for AI Analysis
        </h4>
        <p style="color: #bf360c; margin: 5px 0 0 0; text-align: center; font-size: 14px;">
            Supports images (JPG, PNG, WebP) and PDF documents
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        help="Upload bills, invoices, or receipts for AI analysis. PDFs will be converted to images automatically.",
        key="bill_upload",
        label_visibility="collapsed"
    )

    if uploaded_file:
        file_ext = uploaded_file.name.lower().split('.')[-1]
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

        # Enhanced file preview with better styling
        col1, col2 = st.columns([1, 2])

        with col1:
            if file_ext == 'pdf':
                st.markdown("""
                <div style="background: #f0f2f6; border-radius: 10px; padding: 15px;
                           text-align: center; border: 2px solid #e0e4e7;">
                    <div style="font-size: 48px; margin-bottom: 10px;">📄</div>
                    <div style="font-weight: 600; color: #1f2937;">PDF Document</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.image(uploaded_file, use_column_width=True, caption="")

        with col2:
            st.markdown(f"### ✅ File Ready for Analysis")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**Type:** {file_ext.upper()}")
            st.markdown(f"**Size:** {file_size_mb:.2f} MB")

            if file_ext == 'pdf':
                st.info("📊 PDF will be converted to images for analysis. Processing may take a moment for multi-page documents.")
            else:
                st.success("🖼️ Image ready for instant analysis!")

            st.markdown("---")
            st.markdown("*💡 Tip: Ask questions like 'Analyze this bill' or 'Extract the total amount'*")

    # Enhanced suggestion chips with better styling
    st.markdown("### 💡 Quick Actions")
    suggestions = [
        "📊 Show me sales summary for this month",
        "👥 Who are the top 5 customers by revenue?",
        "📄 Analyze this bill/invoice",
        "🔍 What is the return policy for electronics?",
        "⚠️ Show me all overdue invoices",
        "📦 What products are low on stock?",
        "💰 What payment methods are accepted?",
        "📋 Show customer balances",
        "📈 Generate monthly sales report",
        "🔍 Search for customer by name",
    ]

    # Create a grid layout for suggestions
    cols = st.columns(3)
    for i, sug in enumerate(suggestions):
        col_idx = i % 3
        if cols[col_idx].button(
            sug,
            key=f"sug_{i}",
            use_container_width=True,
            help=f"Click to ask: {sug}"
        ):
            if "Analyze this bill" in sug and uploaded_file:
                st.session_state["prefill_chat"] = "Please analyze this uploaded bill/invoice and extract all the key information including amounts, dates, items, and payment details."
            else:
                st.session_state["prefill_chat"] = sug

    st.divider()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                meta = msg["meta"]
                html = ""
                if meta.get("route"):
                    html += f'<span class="agent-badge">🧭 {meta["route"]}</span>'
                for t in meta.get("tools_used", []):
                    html += f'<span class="tool-chip">🔧 {t}</span>'
                if html:
                    st.markdown(html, unsafe_allow_html=True)

    # Chat input
    prefill = st.session_state.pop("prefill_chat", "")
    user_input = st.chat_input(
        "Ask about sales, customers, policies, bills, or anything else...",
        key="chat_main"
    ) or prefill

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Prepare file data if uploaded (image or PDF)
        file_base64 = None
        if uploaded_file:
            file_bytes = uploaded_file.read()
            if len(file_bytes) <= 10 * 1024 * 1024:  # 10MB limit
                file_base64 = base64.b64encode(file_bytes).decode()
            else:
                st.error("File too large (max 10MB)")

        # Run agent with progress indicator
        with st.chat_message("assistant"):
            if uploaded_file and file_ext == 'pdf':
                with st.spinner("🔄 Converting PDF to images and analyzing... This may take a moment."):
                    result = run_agent(
                        user_message=user_input,
                        history=st.session_state.chat_history[:-1],  # exclude current message
                        file_base64=file_base64
                    )
            else:
                with st.spinner("🤖 Thinking..."):
                    result = run_agent(
                        user_message=user_input,
                        history=st.session_state.chat_history[:-1],  # exclude current message
                        file_base64=file_base64
                    )

            st.markdown(result["response"])

            # Show metadata
            html = ""
            if result.get("route"):
                html += f'<span class="agent-badge">🧭 {result["route"]}</span>'
            for t in result.get("tools_used", []):
                html += f'<span class="tool-chip">🔧 {t}</span>'
            if html:
                st.markdown(html, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"],
            "meta": {"route": result.get("route"), "tools_used": result.get("tools_used", [])},
        })

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()



# ═════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Administration")

    # ── RAG / Vector Store management ─────────────────────────────
    st.subheader("📚 RAG Vector Store")

    from rag.rag_pipeline import is_indexed

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Current status:**")
        if is_indexed():
            st.success("✅ Vector index exists and is ready")
        else:
            st.warning("⚠️ Vector index not built")

    with col2:
        # Rebuild index
        if st.button("🔄 Rebuild Vector Index", type="primary"):
            with st.spinner("Indexing T&C documents..."):
                from rag.rag_pipeline import index_documents
                count = index_documents()
            st.success(f"✅ Rebuilt index with {count} chunks from T&C documents")
            st.rerun()

        # CLEAR EMBEDDINGS BUTTON
        if st.button("🗑️ Clear Vector Embeddings", type="secondary"):
            from rag.rag_pipeline import clear_vector_store
            msg = clear_vector_store()
            st.info(msg)
            st.rerun()

    st.divider()

    # ── LangSmith status ──────────────────────────────────────────
    st.subheader("🔍 LangSmith Tracing")
    key = os.getenv("LANGCHAIN_API_KEY","")
    tracing = os.getenv("LANGCHAIN_TRACING_V2","false")
    project = os.getenv("LANGCHAIN_PROJECT","shop-accounts-capstone")

    col1, col2 = st.columns(2)
    col1.markdown(f"**Tracing enabled:** `{tracing}`")
    col1.markdown(f"**Project:** `{project}`")

    if key and not key.startswith("ls__your"):
        col2.success("✅ LangSmith key configured")
        col2.markdown("[View traces →](https://smith.langchain.com/)")
    else:
        col2.warning("⚠️ LangSmith key not configured")
        col2.markdown("Add `LANGCHAIN_API_KEY=ls__...` to your `.env` file")

    st.divider()

    # ── Database info ─────────────────────────────────────────────
    st.subheader("🗄️ Database")
    from db.database import engine, get_table_schema
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        tables = conn.execute(
            sa_text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).fetchall()

    st.markdown(f"**Database:** `{os.getenv('DATABASE_URL')}`")
    st.markdown("**Tables:**")
    col1, col2 = st.columns(2)
    for i, (tbl,) in enumerate(tables):
        with engine.connect() as conn:
            count = conn.execute(sa_text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
        (col1 if i % 2 == 0 else col2).markdown(f"- `{tbl}` — **{count}** rows")

    with st.expander("📋 Full Schema"):
        st.code(get_table_schema())

    st.divider()

    # ── LangGraph architecture diagram ────────────────────────────
    st.subheader("🧠 LangGraph Architecture")
    st.markdown("""
```
                    ┌─────────────────────────────────────────────────┐
                    │               LANGGRAPH AGENT GRAPH              │
                    └─────────────────────────────────────────────────┘

  User Input ──► [GUARDRAILS] ──► [SUPERVISOR NODE]
                                          │
                    ┌─────────────────────┼──────────────────────┐
                    │           │          │          │           │
                    ▼           ▼          ▼          ▼           ▼
              [sql_agent] [rag_agent] [bill_agent] [analytics] [general]
                    │           │          │          │           │
                    └─────────────────────┼──────────────────────┘
                                          │
                                   [TOOL NODE]  ◄── MCP Tools:
                                          │         • query_database
                                          │         • search_policies
                                          │         • analyze_bill_image
                                          │         • get_customer_summary
                                          │         • get_sales_summary
                                          │         • check_overdue_invoices
                                          │         • get_low_stock_alerts
                                          │
                                   [FINAL ANSWER]
                                          │
                              [GUARDRAILS: sanitize output]
                                          │
                                     User sees response
```
""")
