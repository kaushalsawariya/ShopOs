"""
mcp/tools.py
------------
MCP Tools defined as LangChain @tool functions.
These are the "capabilities" registered with the LangGraph agent.

Tools available:
  query_database       — Text-to-SQL on shop data
  search_policies      — RAG over T&C documents
  analyze_bill_image   — GPT-4o Vision bill extraction (RAG-augmented)
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
    from db.database import engine, get_table_schema
    from guardrails.guardrails import validate_sql, sanitize_output
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from sqlalchemy import text as sa_text

    schema = get_table_schema()
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                     temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    # Generate SQL
    system = f"""You are an expert SQLite assistant.
Convert the user's question into a single SELECT query.
Return ONLY the SQL — no markdown, no backticks, no explanation.

DATABASE SCHEMA:
{schema}

RULES:
- Only SELECT queries
- Use JOINs when needed
- Alias columns clearly
- Limit to 50 rows unless user asks for more
"""
    sql = llm.invoke([SystemMessage(content=system),
                      HumanMessage(content=question)]).content.strip()
    sql = sql.replace("```sql","").replace("```","").strip()

    # Guardrail — validate SQL
    safe, reason = validate_sql(sql)
    if not safe:
        return f"Query blocked by guardrail: {reason}"

    # Execute
    try:
        with engine.connect() as conn:
            rows = conn.execute(sa_text(sql)).fetchall()
            if not rows:
                return f"SQL: {sql}\nResult: No records found."
            cols = list(rows[0]._fields)
            data = [dict(zip(cols, r)) for r in rows]

        # Generate plain-English explanation
        explanation = llm.invoke([
            HumanMessage(content=
                f"Question: {question}\nSQL: {sql}\nResults (first 5): {data[:5]}\n"
                "Explain results in 2–3 sentences using ₹ for currency."
            )
        ]).content

        return sanitize_output(
            f"SQL: {sql}\n\nRows returned: {len(data)}\n\n"
            f"Explanation: {explanation}\n\n"
            f"Data: {json.dumps(data[:20], default=str)}"
        )
    except Exception as e:
        return f"Query execution error: {e}\nSQL attempted: {sql}"


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
    from rag.rag_pipeline import answer_rag_query
    result = answer_rag_query(query)
    if result["error"]:
        return f"Policy search error: {result['error']}"
    sources_str = ", ".join(result["sources"]) if result["sources"] else "Shop policies"
    return f"Answer: {result['answer']}\n\nSources consulted: {sources_str}"


# ══════════════════════════════════════════════════════════════
# TOOL 3 — Bill / Invoice Image Analysis (RAG-augmented)
# ══════════════════════════════════════════════════════════════
@tool
def analyze_bill_image(image_base64: str, filename: str = "bill.jpg") -> str:
    """
    Analyse a bill or invoice image using GPT-4o Vision.
    RAG context about invoice structure and billing terms is retrieved first
    to make the extraction more accurate.
    Input: base64-encoded image string.
    Returns: structured extracted data + analysis.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from rag.rag_pipeline import get_bill_context
    from guardrails.guardrails import sanitize_output
    import json, pathlib

    # 1. Retrieve billing context from RAG vector store
    bill_context = get_bill_context()

    # 2. Build extraction prompt augmented with RAG context
    extraction_prompt = f"""You are an expert invoice reader for an Indian shop.
Use the billing knowledge below to accurately extract all fields.

BILLING KNOWLEDGE (from shop policy documents):
{bill_context}

Extract ALL fields from the bill image and return a JSON object with:
{{
  "invoice_number": "string or null",
  "vendor_name": "string or null",
  "customer_name": "string or null",
  "date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "items": [{{"description":"","qty":0,"unit_price":0,"amount":0}}],
  "subtotal": 0,
  "cgst": 0,
  "sgst": 0,
  "igst": 0,
  "discount": 0,
  "total_amount": 0,
  "payment_method": "string or null",
  "payment_status": "paid/pending/partial/overdue or null",
  "currency": "INR",
  "notes": "string or null"
}}

Return ONLY the JSON — no markdown, no backticks."""

    vision_llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0, max_tokens=1500,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Determine media type from filename
    ext = pathlib.Path(filename).suffix.lower()
    media_type = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                  "png":"image/png","webp":"image/webp"}.get(ext.lstrip("."), "image/jpeg")

    try:
        msg = HumanMessage(content=[
            {"type": "image_url",
             "image_url": {"url": f"data:{media_type};base64,{image_base64}", "detail": "high"}},
            {"type": "text", "text": extraction_prompt},
        ])
        raw = vision_llm.invoke([msg]).content.strip()
        raw = raw.replace("```json","").replace("```","").strip()
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = {"error": "Could not parse JSON from image", "raw": raw}
    except Exception as e:
        return f"Bill analysis error: {e}"

    # 3. Generate human-readable analysis
    analysis_prompt = f"""Based on this extracted bill data, provide a concise 3–5 sentence analysis:
{json.dumps(extracted, indent=2)}

Cover: what the bill is for, key amounts (subtotal/tax/total), payment status,
and any notable items or concerns. Use ₹ for Indian Rupee amounts."""

    analysis = vision_llm.invoke([HumanMessage(content=analysis_prompt)]).content

    # Save bill image to disk for record keeping
    try:
        pathlib.Path("./data/bills").mkdir(parents=True, exist_ok=True)
        img_bytes = base64.b64decode(image_base64)
        with open(f"./data/bills/{filename}", "wb") as f:
            f.write(img_bytes)
    except Exception:
        pass

    return sanitize_output(
        f"EXTRACTED DATA:\n{json.dumps(extracted, indent=2)}\n\n"
        f"ANALYSIS:\n{analysis}"
    )


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
    analyze_bill_image,
    get_customer_summary,
    get_sales_summary,
    check_overdue_invoices,
    get_low_stock_alerts,
]
