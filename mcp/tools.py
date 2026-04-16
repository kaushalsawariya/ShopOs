"""
mcp/tools.py
------------
MCP Tools defined as LangChain @tool functions.
These are the "capabilities" registered with the LangGraph agent.

Tools available:
  query_database       — Text-to-SQL on shop data
  search_policies      — RAG over T&C documents
  analyze_bill_file    — GPT-4o Vision bill extraction (RAG-augmented, supports images & PDFs)
  get_customer_summary — Full customer profile from DB
  get_sales_summary    — Aggregated sales analytics
  check_overdue_invoices — Overdue/pending invoice list
  get_low_stock_alerts — Products below reorder level

Each tool is decorated with @tool, which LangGraph uses to auto-bind
function signatures as tool schemas — no manual JSON schema needed.
"""

import os, json, base64
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════
# TOOL 1 — Text-to-SQL
# ══════════════════════════════════════════════════════════════
@tool
def query_database(question: str) -> str:
    """
    Convert a natural language question into SQL and query the shop database.
    Use for questions about customers, sales, products, invoices, expenses, suppliers.
    Example: 'What are the top 5 customers by total sales?'
    """
    from core.sql_handler import SQLHandler
    handler = SQLHandler()
    return handler.query(question)


# ══════════════════════════════════════════════════════════════
# TOOL 2 — RAG Policy Search
# ══════════════════════════════════════════════════════════════
@tool
def search_policies(query: str) -> str:
    """
    Search the shop's terms and conditions, return policy, warranty, credit terms,
    and payment policies using semantic search (RAG).
    Use when asked about policies, returns, warranties, payment methods, discounts.
    Example: 'What is the return policy for electronics?'
    """
    from core.rag_handler import RAGHandler
    handler = RAGHandler()
    result = handler.query(query)
    if result["error"]:
        return f"Policy search error: {result['error']}"
    sources_str = ", ".join(result["sources"]) if result["sources"] else "Shop policies"
    return f"Answer: {result['answer']}\n\nSources consulted: {sources_str}"


# ══════════════════════════════════════════════════════════════
# TOOL 3 — Bill / Invoice Analysis (Images & PDFs)
# ══════════════════════════════════════════════════════════════
@tool
def analyze_bill_file(file_base64: str, filename: str = "bill.jpg") -> str:
    """
    Analyse a bill or invoice file (image or PDF) using GPT-4o Vision.
    For PDFs, converts pages to images first, then analyzes the content.
    RAG context about invoice structure and billing terms is retrieved first
    to make the extraction more accurate.
    Input: base64-encoded file string (image or PDF).
    Returns: structured extracted data + analysis.
    """
    from core.bill_analyzer import BillAnalyzer
    analyzer = BillAnalyzer()
    return analyzer.analyze_bill(file_base64, filename)


# ══════════════════════════════════════════════════════════════
# TOOL 4 — Customer Summary
# ══════════════════════════════════════════════════════════════
@tool
def get_customer_summary(customer_name: str) -> str:
    """
    Retrieve a comprehensive profile for a customer: contact info, credit status,
    total purchase history, outstanding balance, and recent invoices.
    Example: 'Get summary for Akash Enterprises'
    """
    from db.database import engine
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        cust = conn.execute(
            sa_text("SELECT * FROM customers WHERE name LIKE :n LIMIT 1"),
            {"n": f"%{customer_name}%"}
        ).fetchone()
        if not cust:
            return f"No customer found matching '{customer_name}'."
        cid = cust[0]

        sales = conn.execute(
            sa_text("SELECT COUNT(*), COALESCE(SUM(net_amount),0) FROM sales WHERE customer_id=:id"),
            {"id": cid}
        ).fetchone()

        invs = conn.execute(
            sa_text("""SELECT invoice_number, amount, paid_amount, status, due_date
                       FROM invoices WHERE customer_id=:id ORDER BY issue_date DESC LIMIT 5"""),
            {"id": cid}
        ).fetchall()

    inv_list = [dict(zip(r._fields, r)) for r in invs]
    return json.dumps({
        "customer": dict(zip(cust._fields, cust)),
        "total_transactions": sales[0],
        "total_revenue": float(sales[1]),
        "recent_invoices": inv_list,
    }, default=str)


# ══════════════════════════════════════════════════════════════
# TOOL 5 — Sales Summary
# ══════════════════════════════════════════════════════════════
@tool
def get_sales_summary(period: str = "all") -> str:
    """
    Get aggregated sales analytics. Period options: today, this_week, this_month, this_year, all.
    Returns total transactions, revenue, average sale, and top payment method.
    """
    from db.database import engine
    from sqlalchemy import text as sa_text

    filters = {
        "today":      "DATE(sale_date) = DATE('now')",
        "this_week":  "sale_date >= DATE('now','-7 days')",
        "this_month": "sale_date >= DATE('now','start of month')",
        "this_year":  "sale_date >= DATE('now','start of year')",
        "all":        "1=1",
    }
    where = filters.get(period, "1=1")

    with engine.connect() as conn:
        row = conn.execute(sa_text(f"""
            SELECT COUNT(*) as txns,
                   COALESCE(SUM(net_amount),0) as revenue,
                   COALESCE(AVG(net_amount),0) as avg_sale,
                   COALESCE(MAX(net_amount),0) as highest
            FROM sales WHERE {where}
        """)).fetchone()

    return json.dumps({
        "period": period, "transactions": row[0],
        "total_revenue": round(float(row[1]),2),
        "avg_sale": round(float(row[2]),2),
        "highest_sale": round(float(row[3]),2),
    })


# ══════════════════════════════════════════════════════════════
# TOOL 6 — Overdue Invoices
# ══════════════════════════════════════════════════════════════
@tool
def check_overdue_invoices() -> str:
    """
    List all overdue and unpaid invoices with customer names and outstanding amounts.
    Use to check which customers have pending payments.
    """
    from db.database import engine
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        rows = conn.execute(sa_text("""
            SELECT c.name, i.invoice_number, i.amount,
                   i.paid_amount, (i.amount-i.paid_amount) AS outstanding, i.due_date, i.status
            FROM invoices i JOIN customers c ON i.customer_id=c.id
            WHERE i.status IN ('overdue','pending') AND i.due_date < DATE('now')
            ORDER BY outstanding DESC
        """)).fetchall()

    data = [dict(zip(r._fields, r)) for r in rows]
    total_outstanding = sum(float(r.get("outstanding",0)) for r in data)
    return json.dumps({"count": len(data), "total_outstanding": round(total_outstanding,2),
                       "invoices": data}, default=str)


# ══════════════════════════════════════════════════════════════
# TOOL 7 — Low Stock Alerts
# ══════════════════════════════════════════════════════════════
@tool
def get_low_stock_alerts() -> str:
    """
    List products at or below their reorder level.
    Use to check which items need to be restocked.
    """
    from db.database import engine
    from sqlalchemy import text as sa_text

    with engine.connect() as conn:
        rows = conn.execute(sa_text("""
            SELECT name, sku, category, stock_quantity, reorder_level,
                   (reorder_level-stock_quantity) AS units_needed
            FROM products
            WHERE stock_quantity <= reorder_level AND active=1
            ORDER BY units_needed DESC
        """)).fetchall()

    data = [dict(zip(r._fields, r)) for r in rows]
    return json.dumps({"low_stock_count": len(data), "products": data})


# ══════════════════════════════════════════════════════════════
# All tools exported for LangGraph binding
# ══════════════════════════════════════════════════════════════
ALL_TOOLS = [
    query_database,
    search_policies,
    analyze_bill_file,
    get_customer_summary,
    get_sales_summary,
    check_overdue_invoices,
    get_low_stock_alerts,
]
