# 🏪 ShopOS — Accounts Management System

> AI-powered shop management: Text-to-SQL · Bill Scanning (Vision + RAG) · Policy Search · Multi-Agent Chat

---

## 📖 What This App Does

ShopOS is a Gen AI capstone project that integrates every major LLM technique:

| Feature | Technology Used |
|---|---|
| Natural language → SQL queries | LangChain + OpenAI GPT-4o |
| Bill/invoice image analysis | GPT-4o Vision + **RAG context** |
| Policy & T&C search | FAISS vector store + embeddings (RAG) |
| Conversational AI assistant | LangGraph multi-agent supervisor |
| Tool execution (MCP pattern) | LangChain `@tool` decorators |
| Tracing & observability | LangSmith |
| Input/output safety | Custom guardrails (injection, SQL, PII) |
| Frontend | Streamlit |
| Database | SQLite + SQLAlchemy |

---

## 🗂️ Project Structure

```
shop_accounts/
├── agents/
│   └── graph.py          # LangGraph multi-agent supervisor graph
├── core/
│   └── tracing.py        # LangSmith setup
├── db/
│   ├── models.py          # SQLAlchemy ORM models
│   ├── database.py        # Engine, session, schema helpers
│   └── seed.py            # Test data (10 customers, 20 products, 35 sales, T&C docs)
├── guardrails/
│   └── guardrails.py      # Input validation, SQL safety, PII redaction
├── mcp/
│   └── tools.py           # MCP tools as LangChain @tool functions
├── rag/
│   └── rag_pipeline.py    # FAISS indexing, search, RAG chain + bill context
├── ui/
│   └── app.py             # Streamlit frontend (8 pages)
├── data/                  # Auto-created: SQLite DB, vector store, bill images
├── .env.example           # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🤖 LangGraph Architecture

```
User Input → [Guardrails] → [SUPERVISOR] → specialist agent → [Tool Node] → Response

Supervisor routes to:
  sql_agent        → query_database, get_customer_summary
  rag_agent        → search_policies
  bill_agent       → analyze_bill_image (Vision + RAG)
  analytics_agent  → get_sales_summary, check_overdue_invoices, get_low_stock_alerts
  general_agent    → all tools, general chat
```

---

## 🛠️ MCP Tools (7 tools)

| Tool | Description |
|---|---|
| `query_database` | Natural language → SQL → execute → explain |
| `search_policies` | Semantic RAG search over T&C documents |
| `analyze_bill_image` | Vision extraction + RAG bill context |
| `get_customer_summary` | Full customer profile + invoices |
| `get_sales_summary` | Aggregated sales analytics by period |
| `check_overdue_invoices` | Overdue invoice list + totals |
| `get_low_stock_alerts` | Products below reorder level |

---

## 📊 Database Schema

| Table | Description |
|---|---|
| `customers` | Name, email, phone, city, credit_limit, outstanding_balance, status |
| `products` | SKU, category, unit_price, cost_price, stock_quantity, reorder_level |
| `sales` | Customer, date, total, discount, tax, net_amount, payment_method, status |
| `sale_items` | Sale line items: product, quantity, unit_price, subtotal |
| `invoices` | Invoice number, customer, issue_date, due_date, amount, paid, status |
| `suppliers` | Supplier master data |
| `expenses` | Category, description, amount, date |
| `terms_conditions` | Policy documents (indexed into RAG vector store) |

---

## ⚙️ Setup

### 1. Clone and install
```bash
cd shop_accounts
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the app
```bash
streamlit run ui/app.py
```

The app will:
- Auto-create the SQLite database
- Seed test data (customers, products, sales, T&C docs)
- Build the FAISS RAG vector index
- Launch Streamlit on http://localhost:8501

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key |
| `OPENAI_MODEL` | No | Default: `gpt-4o` |
| `LANGCHAIN_API_KEY` | No | LangSmith key for tracing |
| `LANGCHAIN_TRACING_V2` | No | Set `true` to enable LangSmith |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `DATABASE_URL` | No | Default: `sqlite:///./data/shop_accounts.db` |
| `VECTOR_STORE_PATH` | No | Default: `./data/vector_store` |

---

## 💻 UI Pages

| Page | What You Can Do |
|---|---|
| **AI Assistant** | Chat with multi-agent system; see routing + tools used |
| **Text-to-SQL** | Type any question → see generated SQL + table results |
| **Bill Scanner** | Upload invoice image → GPT-4o Vision + RAG extraction |
| **Policy Search** | Ask about return policy, warranty, credit terms (RAG) |
| **Customers** | Browse all customers with search and filter |
| **Sales** | Browse sales with payment status and method filters |
| **Dashboard** | KPIs, monthly revenue chart, quick action buttons |
| **Settings** | Rebuild or **Clear vector embeddings**, DB schema, LangSmith status |

---

## 🧪 Sample Inputs & Expected Outputs

### Text-to-SQL
```
Input:  "Show top 5 customers by total sales"
SQL:    SELECT c.name, SUM(s.net_amount) AS total FROM sales s JOIN customers...
Output: Table with 5 rows + explanation in plain English
```

### Bill Scanner
```
Input:  JPEG image of a shop bill
Output: {invoice_number, vendor, date, line items, subtotal, GST, total, status}
        + 3-sentence analysis using RAG bill-reading context
```

### Policy Search (RAG)
```
Input:  "What happens if I pay late?"
Output: Answer citing credit account terms + overdue charges policy
        Sources: [Credit Account Terms]
```

### AI Assistant
```
Input:  "Who owes us the most money?"
Route:  supervisor → sql_agent → query_database tool
Output: Ranked list of customers by outstanding balance
```

---

## 🛡️ Guardrails

- **Input validation**: Length check, empty check
- **Prompt injection detection**: Blocks patterns like "ignore previous instructions"
- **SQL safety**: Only SELECT allowed; blocks DROP/DELETE/ALTER etc.
- **PII redaction**: Credit card numbers, Aadhar, PAN auto-redacted from output
- **File type validation**: Bill scanner only accepts image MIME types

---

## 📚 RAG Details

### Indexed Documents
1. Return and Refund Policy
2. Credit Account Terms
3. Warranty Policy
4. Payment and Pricing Policy
5. Bill and Invoice Reading Guide ← used to augment bill image analysis

### Bill Analysis Flow (RAG-augmented)
```
1. User uploads bill image
2. RAG retrieves billing context (invoice structure, abbreviations, GST rules)
3. Extraction prompt = GPT-4o Vision + retrieved context
4. Result: more accurate field extraction + policy-aware analysis
```

### Clear Embeddings
Go to **Settings → Clear Vector Embeddings** to delete the FAISS index.
Use **Rebuild Vector Index** to re-index from the current T&C database records.

---

## 🔬 LangSmith Tracing

Every LLM call is traced when `LANGCHAIN_TRACING_V2=true`:
- Supervisor routing decisions
- Tool calls and results
- RAG retrieval + generation
- SQL generation and explanation

View traces at [smith.langchain.com](https://smith.langchain.com/)

---

## 📦 Test Data Summary

- 10 customers (mix of VIP, active, inactive; cities: Delhi, Gurugram, Noida, Faridabad)
- 20 products (Electronics, Furniture, Stationery, Health & Safety)
- 5 suppliers
- 35 sales transactions (2024, various payment methods and statuses)
- 25+ invoices (mix of paid, pending, overdue, partial)
- 60 expense records (6 months × 10 categories)
- 5 T&C/policy documents (indexed into RAG)
