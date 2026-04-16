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
  :root {
    --bg: #f4efe6;
    --panel: rgba(255,255,255,0.84);
    --ink: #1f2a2e;
    --muted: #617076;
    --line: rgba(80, 63, 43, 0.14);
    --accent: #b85c38;
    --accent-soft: rgba(184, 92, 56, 0.12);
    --accent-2: #2f6c7a;
    --accent-2-soft: rgba(47, 108, 122, 0.12);
    --shadow: 0 18px 40px rgba(94, 73, 44, 0.10);
  }
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(184, 92, 56, 0.10), transparent 28%),
      radial-gradient(circle at top right, rgba(47, 108, 122, 0.12), transparent 22%),
      linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
    color: var(--ink);
    font-family: "Aptos", "Trebuchet MS", "Segoe UI", sans-serif;
  }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,250,242,0.95) 0%, rgba(243,235,223,0.95) 100%);
    border-right: 1px solid var(--line);
  }
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  .stButton>button {
    border-radius: 999px; font-weight: 700; transition: all .18s ease;
    background: linear-gradient(180deg, #fffdf9 0%, #f8efe3 100%);
    border: 1px solid rgba(184, 92, 56, 0.25); color: var(--ink);
    box-shadow: 0 8px 18px rgba(84, 56, 27, 0.08);
  }
  .stButton>button:hover {
    border-color: rgba(184, 92, 56, 0.55) !important;
    color: var(--accent) !important; transform: translateY(-1px);
  }
  [data-testid="stFileUploader"], [data-testid="stChatMessage"] {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 20px; box-shadow: var(--shadow);
  }
  [data-testid="stFileUploader"] { padding: 0.4rem 0.8rem; }
  [data-testid="stChatMessage"] { padding: 0.7rem 0.85rem; margin-bottom: 0.8rem; }
  .hero-card {
    background:
      radial-gradient(circle at 85% 15%, rgba(255,255,255,0.5), transparent 20%),
      linear-gradient(135deg, #204e57 0%, #2f6c7a 42%, #e9dcc8 42%, #f8f2e8 100%);
    border: 1px solid rgba(32, 78, 87, 0.12);
    border-radius: 28px; padding: 26px 28px; margin: 0 0 18px 0;
    box-shadow: var(--shadow); overflow: hidden;
  }
  .hero-kicker {
    display: inline-block; padding: 6px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.16); color: #fff8ef;
    border: 1px solid rgba(255,255,255,0.2); font-size: 11px;
    letter-spacing: .14em; text-transform: uppercase; font-weight: 800;
  }
  .hero-title {
    margin: 14px 0 8px 0; font-size: 2.25rem; line-height: 1.02;
    font-weight: 900; color: #fffaf1; max-width: 560px;
  }
  .hero-copy { margin: 0; max-width: 560px; color: rgba(255,248,238,0.86); font-size: 1rem; line-height: 1.55; }
  .stat-grid, .feature-list, .settings-grid {
    display: grid; gap: 12px;
  }
  .stat-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 20px; }
  .feature-list, .settings-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 14px; }
  .stat-card, .surface-card, .sidebar-panel, .settings-tile, .feature-pill {
    background: var(--panel); border: 1px solid var(--line);
    box-shadow: 0 10px 24px rgba(92, 71, 46, 0.06);
  }
  .stat-card { border-radius: 18px; padding: 14px 16px; }
  .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: rgba(32,45,48,0.62); margin-bottom: 4px; font-weight: 700; }
  .stat-value { font-size: 1.1rem; font-weight: 800; color: var(--ink); }
  .surface-card, .settings-tile { border-radius: 24px; padding: 18px 20px; margin-bottom: 16px; }
  .surface-title { font-size: 1.08rem; font-weight: 800; color: var(--ink); margin-bottom: 4px; }
  .surface-copy, .mini-note, .settings-tile span { color: var(--muted); font-size: 0.95rem; }
  .sidebar-panel { border-radius: 20px; padding: 14px 16px; margin-bottom: 14px; }
  .sidebar-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.14em; color: var(--muted); margin-bottom: 6px; font-weight: 800; }
  .sidebar-big { font-size: 1.05rem; font-weight: 800; color: var(--ink); }
  .feature-pill { border-radius: 16px; padding: 10px 12px; font-size: 0.92rem; color: var(--ink); }
  .agent-badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 11px;
    font-weight: 800; background: var(--accent-soft); color: var(--accent);
    border: 1px solid rgba(184, 92, 56, 0.18); margin-right: 6px; margin-top: 6px;
  }
  .tool-chip {
    display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 10px;
    font-weight: 700; background: var(--accent-2-soft); color: var(--accent-2);
    border: 1px solid rgba(47,108,122,0.16); margin: 6px 6px 0 0;
  }
  @media (max-width: 900px) {
    .stat-grid, .feature-list, .settings-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.8rem; }
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-card">
      <div class="hero-kicker">ShopOS Control Room</div>
      <div class="hero-title">Run sales questions, policy search, and invoice analysis from one calm workspace.</div>
      <p class="hero-copy">
        Ask in plain English, upload bills or PDFs, and move between shop operations,
        analytics, and document analysis without losing context.
      </p>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-label">Agents</div>
          <div class="stat-value">5 specialist routes</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Knowledge</div>
          <div class="stat-value">SQL + RAG + Vision</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Documents</div>
          <div class="stat-value">Images and PDFs</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("""
    <div class="sidebar-panel">
      <div class="sidebar-title">Workspace</div>
      <div class="sidebar-big">ShopOS</div>
      <div class="mini-note">AI operations assistant for accounts, documents, and policy search.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-panel">
      <div class="sidebar-title">Status</div>
      <div class="sidebar-big">Online and Ready</div>
      <div class="mini-note">Supervisor, analytics, SQL, bill, and RAG routes are available.</div>
    </div>
    """, unsafe_allow_html=True)
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

    render_hero()
    st.markdown("""
    <div class="surface-card">
      <div class="surface-title">Operations chat built for real shop workflows</div>
      <p class="surface-copy">
        Ask about balances, sales, inventory, returns, or uploaded invoices. The assistant routes
        the request to the best specialist instead of forcing everything through one generic path.
      </p>
      <div class="feature-list">
        <div class="feature-pill">Natural language database questions</div>
        <div class="feature-pill">Policy and terms search with RAG</div>
        <div class="feature-pill">Bill analysis from images and PDFs</div>
        <div class="feature-pill">Inventory and overdue invoice monitoring</div>
      </div>
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

    st.markdown("""
    <div class="surface-card">
      <div class="surface-title">Document intake</div>
      <p class="surface-copy">
        Upload a bill, invoice, or receipt to extract key fields and get a short operational summary.
        Images and PDFs are both supported.
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
    st.markdown("""
    <div class="surface-card">
      <div class="surface-title">Quick prompts</div>
      <p class="surface-copy">
        Use one of these shortcuts to jump into a common operational task.
      </p>
    </div>
    """, unsafe_allow_html=True)

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
        file_name = None
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) <= 10 * 1024 * 1024:  # 10MB limit
                file_base64 = base64.b64encode(file_bytes).decode()
                file_name = uploaded_file.name
            else:
                st.error("File too large (max 10MB)")

        # Run agent with progress indicator
        with st.chat_message("assistant"):
            if uploaded_file and file_ext == 'pdf':
                with st.spinner("🔄 Converting PDF to images and analyzing... This may take a moment."):
                    result = run_agent(
                        user_message=user_input,
                        history=st.session_state.chat_history[:-1],  # exclude current message
                        file_base64=file_base64,
                        file_name=file_name
                    )
            else:
                with st.spinner("🤖 Thinking..."):
                    result = run_agent(
                        user_message=user_input,
                        history=st.session_state.chat_history[:-1],  # exclude current message
                        file_base64=file_base64,
                        file_name=file_name
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

    st.markdown("""
    <div class="hero-card">
      <div class="hero-kicker">System Desk</div>
      <div class="hero-title">Settings, diagnostics, and knowledge-store maintenance.</div>
      <p class="hero-copy">
        Check the vector index, review tracing status, inspect the database, and manage the
        operational scaffolding behind the assistant.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="settings-grid">
      <div class="settings-tile">
        <strong>Vector store controls</strong>
        <span>Rebuild or clear embeddings when the knowledge base changes.</span>
      </div>
      <div class="settings-tile">
        <strong>Tracing visibility</strong>
        <span>Check whether LangSmith observability is configured for this environment.</span>
      </div>
      <div class="settings-tile">
        <strong>Database health</strong>
        <span>Inspect available tables, row counts, and current schema details.</span>
      </div>
      <div class="settings-tile">
        <strong>Architecture reference</strong>
        <span>Keep the routing model and tool graph visible while debugging.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
