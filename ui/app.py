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
  /* Dark theme overrides */
  [data-testid="stSidebar"] { background: #141720; border-right: 1px solid #2a3045; }
  .stButton>button { border-radius: 8px; font-weight: 600; transition: all .15s; }
  .stButton>button:hover { border-color: #00e5b0 !important; color: #00e5b0 !important; }
  .metric-card {
    background: #141720; border: 1px solid #2a3045; border-radius: 12px;
    padding: 16px 20px; margin-bottom: 4px;
  }
  .metric-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .8px;
    color: #8a95b0; margin-bottom: 4px; }
  .metric-card .value { font-size: 28px; font-weight: 800; color: #e8ecf5; line-height: 1.1; }
  .metric-card .sub { font-size: 11px; color: #505870; margin-top: 4px; }
  .agent-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; background: rgba(0,229,176,.12);
    color: #00e5b0; border: 1px solid rgba(0,229,176,.25); margin-left: 8px;
  }
  .tool-chip {
    display: inline-block; padding: 1px 8px; border-radius: 20px;
    font-size: 10px; background: #1c2030; color: #8a95b0;
    border: 1px solid #2a3045; margin: 1px;
  }
  .source-chip {
    display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-size: 11px; background: rgba(0,229,176,.08); color: #00e5b0;
    border: 1px solid rgba(0,229,176,.2); margin: 2px;
  }
  .sql-block { background: #0d0f12; border-radius: 8px; padding: 12px;
    font-family: 'DM Mono', monospace; font-size: 13px; color: #00e5b0;
    border: 1px solid #2a3045; white-space: pre-wrap; word-break: break-all; }
  .bill-field { background: #1c2030; border-radius: 6px; padding: 10px 14px; margin: 4px 0; }
  .bill-field .lbl { font-size: 10px; text-transform: uppercase; color: #505870; }
  .bill-field .val { font-size: 14px; color: #e8ecf5; font-family: monospace; }
  .warning-box { background: rgba(255,179,64,.08); border: 1px solid rgba(255,179,64,.3);
    border-radius: 8px; padding: 12px; color: #ffb340; font-size: 13px; }
  hr { border-color: #2a3045 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏪 ShopOS")
    st.markdown("*Accounts Management System*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🤖 AI Assistant", "🗄️ Text-to-SQL", "📄 Bill Scanner",
         "🛡️ Policy Search", "👥 Customers", "📈 Sales", "📊 Dashboard", "⚙️ Settings"],
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
    st.caption("Multi-agent chat powered by LangGraph · Supervisor → Specialist Agents → MCP Tools")

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggestion chips
    suggestions = [
        "Who are the top 5 customers by revenue?",
        "What is the return policy for electronics?",
        "Show me all overdue invoices",
        "What products are low on stock?",
        "Summarise sales for this year",
    ]
    st.markdown("**Quick start:**")
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}", use_container_width=True):
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
        "Ask about sales, customers, policies, or upload a bill...",
        key="chat_main"
    ) or prefill

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from agents.graph import run_agent
                result = run_agent(
                    user_message=user_input,
                    history=st.session_state.chat_history[:-1],  # exclude current message
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
# PAGE: TEXT-TO-SQL
# ═════════════════════════════════════════════════════════════════
elif page == "🗄️ Text-to-SQL":
    st.title("🗄️ Text-to-SQL Query Engine")
    st.caption("Write questions in plain English — get SQL + results + explanation")

    # Example queries
    examples = {
        "Top 5 customers by revenue": "Show me the top 5 customers ranked by total net sales amount",
        "Low stock electronics": "List all Electronics products with stock quantity below 50",
        "Overdue invoices": "Show all overdue invoices with customer name and outstanding amount",
        "Monthly sales 2024": "What is the total sales revenue for each month in 2024?",
        "Payment method breakdown": "Show count and total value of sales grouped by payment method",
        "VIP customer balances": "List all VIP customers with their outstanding balance and credit limit",
    }

    st.markdown("**Example queries:**")
    cols = st.columns(3)
    for i, (label, query) in enumerate(examples.items()):
        if cols[i % 3].button(label, key=f"ex_{i}"):
            st.session_state["sql_prefill"] = query

    question = st.text_area(
        "Your question",
        value=st.session_state.pop("sql_prefill", ""),
        placeholder="e.g. Show total revenue per customer this year...",
        height=80,
    )

    if st.button("▶ Run Query", type="primary") and question.strip():
        from guardrails.guardrails import run_guardrails, validate_sql, sanitize_output
        from db.database import engine, get_table_schema
        from langchain_openai import ChatOpenAI
        from langchain.schema import SystemMessage, HumanMessage
        from sqlalchemy import text as sa_text

        passed, reason = run_guardrails(question)
        if not passed:
            st.error(f"⚠️ Guardrail blocked: {reason}")
        else:
            with st.spinner("Generating SQL..."):
                schema = get_table_schema()
                llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL","gpt-4o"),
                                 temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
                sys_msg = f"""You are an expert SQLite assistant.
Convert the question to a single SELECT query. Return ONLY the SQL — no markdown.
SCHEMA:
{schema}
Rules: SELECT only, clear aliases, LIMIT 50."""
                sql = llm.invoke([SystemMessage(content=sys_msg),
                                  HumanMessage(content=question)]).content.strip()
                sql = sql.replace("```sql","").replace("```","").strip()

            # Guardrail — SQL safety
            safe, sreason = validate_sql(sql)
            st.markdown("**Generated SQL:**")
            st.markdown(f'<div class="sql-block">{sql}</div>', unsafe_allow_html=True)

            if not safe:
                st.error(f"🚫 SQL blocked by guardrail: {sreason}")
            else:
                with st.spinner("Executing..."):
                    try:
                        with engine.connect() as conn:
                            rows = conn.execute(sa_text(sql)).fetchall()

                        st.success(f"✅ {len(rows)} row(s) returned")

                        if rows:
                            cols_names = list(rows[0]._fields)
                            df = pd.DataFrame([dict(zip(cols_names, r)) for r in rows])
                            st.dataframe(df, use_container_width=True, height=400)

                            # LLM explanation
                            with st.spinner("Generating explanation..."):
                                expl = llm.invoke([HumanMessage(content=
                                    f"Question: {question}\nSQL: {sql}\n"
                                    f"Results (first 5): {df.head().to_dict()}\n"
                                    "Explain in 2–3 sentences using ₹ for currency."
                                )]).content
                            st.info(f"💡 {sanitize_output(expl)}")
                        else:
                            st.warning("No records found for this query.")
                    except Exception as e:
                        st.error(f"Query error: {e}")


# ═════════════════════════════════════════════════════════════════
# PAGE: BILL SCANNER
# ═════════════════════════════════════════════════════════════════
elif page == "📄 Bill Scanner":
    st.title("📄 Bill / Invoice Scanner")
    st.caption("Upload a bill image → GPT-4o Vision + RAG context → Structured extraction")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload bill image",
            type=["jpg","jpeg","png","webp"],
            help="Supports JPEG, PNG, WebP up to 10MB"
        )

        if uploaded:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

            # Show RAG context being used
            with st.expander("📚 RAG context loaded for bill analysis"):
                from rag.rag_pipeline import get_bill_context
                ctx = get_bill_context()
                st.text(ctx[:800] + "..." if len(ctx) > 800 else ctx)

            if st.button("🔍 Analyse Bill", type="primary"):
                img_bytes = uploaded.read()
                if len(img_bytes) > 10 * 1024 * 1024:
                    st.error("File too large (max 10MB)")
                else:
                    with st.spinner("🤖 Analysing with GPT-4o Vision + RAG..."):
                        from mcp.tools import analyze_bill_image as _analyze
                        img_b64 = base64.b64encode(img_bytes).decode()
                        raw = _analyze.invoke({
                            "image_base64": img_b64,
                            "filename": uploaded.name,
                        })
                        st.session_state["bill_result"] = raw
                        st.session_state["bill_filename"] = uploaded.name

    with col2:
        if "bill_result" in st.session_state:
            raw = st.session_state["bill_result"]

            # Parse extracted JSON section
            if "EXTRACTED DATA:" in raw:
                parts = raw.split("ANALYSIS:")
                extracted_section = parts[0].replace("EXTRACTED DATA:","").strip()
                analysis_section  = parts[1].strip() if len(parts) > 1 else ""

                st.markdown("### 📊 Analysis")
                st.info(analysis_section)

                st.markdown("### 🗂️ Extracted Fields")
                try:
                    data = json.loads(extracted_section)

                    # Display key fields
                    fields = [
                        ("Invoice Number", data.get("invoice_number")),
                        ("Vendor",         data.get("vendor_name")),
                        ("Customer",       data.get("customer_name")),
                        ("Date",           data.get("date")),
                        ("Due Date",       data.get("due_date")),
                        ("Subtotal",       f"₹{data.get('subtotal',0):,.2f}" if data.get("subtotal") else None),
                        ("CGST",           f"₹{data.get('cgst',0):,.2f}" if data.get("cgst") else None),
                        ("SGST",           f"₹{data.get('sgst',0):,.2f}" if data.get("sgst") else None),
                        ("Discount",       f"₹{data.get('discount',0):,.2f}" if data.get("discount") else None),
                        ("Total Amount",   f"₹{data.get('total_amount',0):,.2f}" if data.get("total_amount") else None),
                        ("Payment Method", data.get("payment_method")),
                        ("Status",         data.get("payment_status")),
                    ]

                    c1, c2 = st.columns(2)
                    for i, (lbl, val) in enumerate(fields):
                        if val:
                            col = c1 if i % 2 == 0 else c2
                            col.markdown(
                                f'<div class="bill-field"><div class="lbl">{lbl}</div>'
                                f'<div class="val">{val}</div></div>',
                                unsafe_allow_html=True
                            )

                    # Line items table
                    items = data.get("items", [])
                    if items:
                        st.markdown("### 🧾 Line Items")
                        df_items = pd.DataFrame(items)
                        st.dataframe(df_items, use_container_width=True)

                except json.JSONDecodeError:
                    st.text(extracted_section)
            else:
                # Fallback: show raw
                st.markdown(raw)


# ═════════════════════════════════════════════════════════════════
# PAGE: POLICY SEARCH (RAG)
# ═════════════════════════════════════════════════════════════════
elif page == "🛡️ Policy Search":
    st.title("🛡️ Policy & Terms Search")
    st.caption("Semantic search over T&C documents using FAISS vector store + OpenAI embeddings")

    # RAG index status
    from rag.rag_pipeline import is_indexed
    if is_indexed():
        st.success("✅ Vector index is ready")
    else:
        st.warning("⚠️ Vector index not built. Go to Settings to rebuild.")

    # Suggested queries
    suggestions = [
        "What is the return policy for electronics?",
        "How long is the warranty on furniture?",
        "What happens if a customer pays late?",
        "What payment methods are accepted?",
        "Can customers get a discount on bulk orders?",
        "What does an invoice contain?",
        "What does CGST mean on a bill?",
    ]
    st.markdown("**Try these:**")
    for sug in suggestions:
        if st.button(f"💬 {sug}", key=f"rag_{sug}"):
            st.session_state["rag_prefill"] = sug

    query = st.text_input(
        "Search policies",
        value=st.session_state.pop("rag_prefill", ""),
        placeholder="Ask about return policy, warranty, credit terms...",
    )

    top_k = st.slider("Number of context chunks to retrieve", 1, 8, 3)

    if st.button("🔍 Search", type="primary") and query.strip():
        from guardrails.guardrails import run_guardrails
        passed, reason = run_guardrails(query)
        if not passed:
            st.error(f"⚠️ Guardrail blocked: {reason}")
        else:
            with st.spinner("Searching policy documents..."):
                from rag.rag_pipeline import answer_rag_query
                result = answer_rag_query(query, k=top_k)

            if result["error"]:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown("### 💡 Answer")
                st.markdown(result["answer"])

                st.markdown("**Sources used:**")
                src_html = "".join(f'<span class="source-chip">📄 {s}</span>'
                                   for s in result["sources"])
                st.markdown(src_html, unsafe_allow_html=True)

                st.caption(f"Retrieved {result['chunks_used']} context chunk(s)")

                # Show retrieved chunks
                with st.expander("📖 Retrieved context chunks"):
                    from rag.rag_pipeline import search
                    docs = search(query, k=top_k)
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}** — `{doc.metadata.get('title','')}`")
                        st.text(doc.page_content[:400])
                        st.divider()


# ═════════════════════════════════════════════════════════════════
# PAGE: CUSTOMERS
# ═════════════════════════════════════════════════════════════════
elif page == "👥 Customers":
    st.title("👥 Customer Directory")

    from db.database import engine
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        rows = conn.execute(sa_text("SELECT * FROM customers ORDER BY name")).fetchall()

    if rows:
        df = pd.DataFrame([dict(zip(r._fields, r)) for r in rows])

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", len(df))
        c2.metric("VIP Customers", len(df[df.status=="vip"]))
        c3.metric("Active", len(df[df.status=="active"]))
        total_outstanding = df.outstanding_balance.sum()
        c4.metric("Total Outstanding", f"₹{total_outstanding:,.0f}")

        st.divider()

        # Search
        search_q = st.text_input("🔍 Search customers", placeholder="Name, city, email...")
        if search_q:
            mask = df.apply(lambda r: search_q.lower() in str(r).lower(), axis=1)
            df = df[mask]

        # Format
        display_cols = ["name","email","phone","city","status","credit_limit","outstanding_balance","joined_date"]
        df_show = df[display_cols].copy()
        df_show["credit_limit"] = df_show["credit_limit"].apply(lambda x: f"₹{x:,.0f}")
        df_show["outstanding_balance"] = df_show["outstanding_balance"].apply(lambda x: f"₹{x:,.0f}")
        df_show.columns = ["Name","Email","Phone","City","Status","Credit Limit","Outstanding","Joined"]

        st.dataframe(df_show, use_container_width=True, height=500)
    else:
        st.info("No customer data. Run seed.py first.")


# ═════════════════════════════════════════════════════════════════
# PAGE: SALES
# ═════════════════════════════════════════════════════════════════
elif page == "📈 Sales":
    st.title("📈 Sales Records")

    from db.database import engine
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        rows = conn.execute(sa_text("""
            SELECT s.id, c.name as customer, s.sale_date,
                   s.total_amount, s.discount, s.tax, s.net_amount,
                   s.payment_method, s.payment_status
            FROM sales s JOIN customers c ON s.customer_id=c.id
            ORDER BY s.sale_date DESC
        """)).fetchall()

    if rows:
        df = pd.DataFrame([dict(zip(r._fields, r)) for r in rows])

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", len(df))
        c2.metric("Total Revenue", f"₹{df.net_amount.sum():,.0f}")
        c3.metric("Average Sale", f"₹{df.net_amount.mean():,.0f}")
        c4.metric("Pending Payments", len(df[df.payment_status=="pending"]))

        st.divider()

        # Filter
        col1, col2 = st.columns(2)
        status_filter = col1.selectbox("Payment Status", ["All","paid","pending","partial"])
        method_filter = col2.selectbox("Payment Method", ["All","cash","card","upi","bank_transfer","credit"])

        if status_filter != "All":
            df = df[df.payment_status == status_filter]
        if method_filter != "All":
            df = df[df.payment_method == method_filter]

        # Format
        df["net_amount"] = df["net_amount"].apply(lambda x: f"₹{x:,.2f}")
        df["total_amount"] = df["total_amount"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.info("No sales data found.")


# ═════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Shop Dashboard")

    from db.database import engine
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        revenue  = conn.execute(sa_text("SELECT COALESCE(SUM(net_amount),0) FROM sales")).scalar()
        customers= conn.execute(sa_text("SELECT COUNT(*) FROM customers")).scalar()
        overdue  = conn.execute(sa_text(
            "SELECT COUNT(*), COALESCE(SUM(amount-paid_amount),0) FROM invoices "
            "WHERE status IN ('overdue','pending') AND due_date < DATE('now')"
        )).fetchone()
        low_stock= conn.execute(sa_text(
            "SELECT COUNT(*) FROM products WHERE stock_quantity<=reorder_level AND active=1"
        )).scalar()
        monthly  = conn.execute(sa_text("""
            SELECT strftime('%Y-%m', sale_date) as month, SUM(net_amount) as rev
            FROM sales WHERE sale_date >= DATE('now','-6 months')
            GROUP BY month ORDER BY month
        """)).fetchall()
        top_custs= conn.execute(sa_text("""
            SELECT c.name, SUM(s.net_amount) as total
            FROM sales s JOIN customers c ON s.customer_id=c.id
            GROUP BY c.name ORDER BY total DESC LIMIT 5
        """)).fetchall()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue",     f"₹{float(revenue):,.0f}")
    c2.metric("👥 Total Customers",   customers)
    c3.metric("⚠️ Overdue Amount",    f"₹{float(overdue[1]):,.0f}", delta=f"{overdue[0]} invoices", delta_color="inverse")
    c4.metric("📦 Low Stock Items",   low_stock, delta_color="inverse")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Monthly Revenue (Last 6 Months)**")
        if monthly:
            df_monthly = pd.DataFrame(monthly, columns=["Month", "Revenue"])
            df_monthly["Revenue"] = df_monthly["Revenue"].astype(float)
            st.bar_chart(df_monthly.set_index("Month"))
        else:
            st.info("No monthly data available")

    with col2:
        st.markdown("**Top 5 Customers by Revenue**")
        if top_custs:
            df_top = pd.DataFrame(top_custs, columns=["Customer", "Revenue"])
            df_top["Revenue"] = df_top["Revenue"].astype(float)
            st.bar_chart(df_top.set_index("Customer"))
        else:
            st.info("No customer data available")

    # Quick action buttons
    st.divider()
    st.markdown("**Quick Actions**")
    qa_cols = st.columns(4)
    if qa_cols[0].button("📋 Overdue Invoices"):
        with engine.connect() as conn:
            rows = conn.execute(sa_text("""
                SELECT c.name, i.invoice_number, i.amount, i.paid_amount,
                       (i.amount-i.paid_amount) as outstanding, i.due_date
                FROM invoices i JOIN customers c ON i.customer_id=c.id
                WHERE i.status IN ('overdue','pending') AND i.due_date<DATE('now')
                ORDER BY outstanding DESC
            """)).fetchall()
        if rows:
            df = pd.DataFrame([dict(zip(r._fields,r)) for r in rows])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No overdue invoices! 🎉")

    if qa_cols[1].button("📦 Low Stock"):
        with engine.connect() as conn:
            rows = conn.execute(sa_text("""
                SELECT name, sku, category, stock_quantity, reorder_level
                FROM products WHERE stock_quantity<=reorder_level AND active=1
                ORDER BY (reorder_level-stock_quantity) DESC
            """)).fetchall()
        if rows:
            df = pd.DataFrame([dict(zip(r._fields,r)) for r in rows])
            st.dataframe(df, use_container_width=True)
        else:
            st.success("All products are well stocked! ✅")

    if qa_cols[2].button("🏆 Top Products"):
        with engine.connect() as conn:
            rows = conn.execute(sa_text("""
                SELECT p.name, SUM(si.quantity) as qty_sold, SUM(si.subtotal) as revenue
                FROM sale_items si JOIN products p ON si.product_id=p.id
                GROUP BY p.name ORDER BY qty_sold DESC LIMIT 10
            """)).fetchall()
        if rows:
            df = pd.DataFrame([dict(zip(r._fields,r)) for r in rows])
            df["revenue"] = df["revenue"].apply(lambda x: f"₹{float(x):,.0f}")
            st.dataframe(df, use_container_width=True)

    if qa_cols[3].button("💳 Payment Split"):
        with engine.connect() as conn:
            rows = conn.execute(sa_text("""
                SELECT payment_method, COUNT(*) as count, SUM(net_amount) as total
                FROM sales GROUP BY payment_method ORDER BY total DESC
            """)).fetchall()
        if rows:
            df = pd.DataFrame([dict(zip(r._fields,r)) for r in rows])
            df["total"] = df["total"].apply(lambda x: f"₹{float(x):,.0f}")
            st.dataframe(df, use_container_width=True)


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
