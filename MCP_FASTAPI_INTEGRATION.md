# MCP + FastAPI Integration Guide

## Overview
Your ShopOs project now has **full MCP integration with FastAPI**. All your existing MCP tools are exposed as REST API endpoints while still being used by LangGraph agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Clients (Streamlit, External Apps, Mobile)                    │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           FastAPI Backend (Port 8000)                   │  │
│  │                                                          │  │
│  │  ┌──────────────┐      ┌──────────────────────────────┐ │  │
│  │  │ /chat        │      │ /mcp/* (MCP Tools)          │ │  │
│  │  │ (Supervisor) │      │ - /mcp/sql-query            │ │  │
│  │  └──────┬───────┘      │ - /mcp/policy-search        │ │  │
│  │         │              │ - /mcp/analyze-bill         │ │  │
│  │         ▼              │ - /mcp/customer-summary     │ │  │
│  │  LangGraph Agents      │ - /mcp/sales-summary        │ │  │
│  │  ├─ SQL Agent          │ - /mcp/overdue-invoices     │ │  │
│  │  ├─ RAG Agent          │ - /mcp/low-stock-alerts     │ │  │
│  │  ├─ Bill Agent         └──────────────────────────────┘ │  │
│  │  ├─ Analytics Agent            │                        │  │
│  │  └─ General Agent              │                        │  │
│  │         │                       │                        │  │
│  │         └───────────┬───────────┘                        │  │
│  │                     ▼                                    │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  MCP Tools (mcp/tools.py)                           │ │  │
│  │  │  - query_database (SQL)                             │ │  │
│  │  │  - search_policies (RAG)                            │ │  │
│  │  │  - analyze_bill_file (Vision)                       │ │  │
│  │  │  - get_customer_summary                             │ │  │
│  │  │  - get_sales_summary                                │ │  │
│  │  │  - check_overdue_invoices                           │ │  │
│  │  │  - get_low_stock_alerts                             │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │         │                                                │  │
│  │         ▼                                                │  │
│  │  Core Systems                                            │  │
│  │  ├─ Database                                             │  │
│  │  ├─ RAG Pipeline                                         │  │
│  │  ├─ LLM (OpenAI)                                          │  │
│  │  └─ Bill Analyzer                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

✅ **MCP Tools Remain Unchanged** - All your existing LangChain tools work as-is  
✅ **FastAPI Wrapping** - Each tool exposed as a REST endpoint  
✅ **CORS Enabled** - Streamlit and external clients can call the API  
✅ **Async Support** - Non-blocking I/O using `asyncio.to_thread()`  
✅ **Automatic Docs** - Swagger UI at `/docs`, ReDoc at `/redoc`  
✅ **Error Handling** - Consistent error responses  
✅ **Tool Metadata** - `/mcp/tools-list` endpoint for discovering available tools  

---

## Available Endpoints

### 1. Agent Routes

#### `POST /chat`
Multi-agent supervisor endpoint. Routes queries to best-fit specialist agent.

**Request:**
```json
{
  "query": "Show me top 5 customers by sales",
  "context": {}
}
```

**Response:**
```json
{
  "response": "Top 5 customers by sales are...",
  "route": "sql_agent",
  "tools_used": ["query_database"]
}
```

---

### 2. MCP Tool Routes

#### `POST /mcp/sql-query`
Text-to-SQL database query tool.

**Request:**
```json
{
  "question": "What are the top 5 customers by total sales?"
}
```

**Response:**
```json
{
  "result": "...",
  "tool": "query_database"
}
```

---

#### `POST /mcp/policy-search`
RAG-based policy and terms search.

**Request:**
```json
{
  "query": "What is the return policy?"
}
```

**Response:**
```json
{
  "result": "Answer: ... Sources: ...",
  "tool": "search_policies"
}
```

---

#### `POST /mcp/analyze-bill`
Bill/Invoice analysis using Vision + RAG.

**Request:**
```json
{
  "file_base64": "iVBORw0KGgoAAAANS...",
  "filename": "invoice.jpg"
}
```

**Response:**
```json
{
  "result": "...",
  "tool": "analyze_bill_file"
}
```

---

#### `POST /mcp/customer-summary`
Get full customer profile.

**Request:**
```json
{
  "customer_name": "Akash Enterprises"
}
```

**Response:**
```json
{
  "result": "...",
  "tool": "get_customer_summary"
}
```

---

#### `GET /mcp/sales-summary`
Aggregated sales analytics.

**Response:**
```json
{
  "result": "...",
  "tool": "get_sales_summary"
}
```

---

#### `GET /mcp/overdue-invoices`
List pending and overdue invoices.

**Response:**
```json
{
  "result": "...",
  "tool": "check_overdue_invoices"
}
```

---

#### `GET /mcp/low-stock-alerts`
Products below reorder level.

**Response:**
```json
{
  "result": "...",
  "tool": "get_low_stock_alerts"
}
```

---

#### `GET /mcp/tools-list`
Discover all available MCP tools.

**Response:**
```json
{
  "tools": [
    {
      "name": "query_database",
      "endpoint": "/mcp/sql-query",
      "method": "POST",
      "description": "Text-to-SQL queries on shop database"
    },
    ...
  ]
}
```

---

## How to Start

### Terminal 1: Start FastAPI Server
```bash
cd c:\Users\Kaushal\OneDrive\Desktop\ShopOs
& c:\Users\Kaushal\OneDrive\Desktop\ShopOs\venv\Scripts\Activate.ps1
uvicorn api.main:app --reload
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Terminal 2: Start Streamlit UI
```bash
streamlit run ui/app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI Docs | http://localhost:8000/docs | Interactive API explorer |
| FastAPI ReDoc | http://localhost:8000/redoc | Alternative API docs |
| Streamlit UI | http://localhost:8501 | Chat + Workflow graph |
| API Base | http://localhost:8000 | REST endpoints |

---

## How to Use MCP with FastAPI

### Option 1: Via Streamlit (Built-in UI)
1. Open http://localhost:8501
2. Go to "🤖 AI Assistant" page
3. Ask questions normally
4. Behind the scenes, it calls the FastAPI `/chat` endpoint
5. View "📊 Workflow Graph" to see agent routing

### Option 2: Direct API Calls (via cURL, Postman, or Python)

#### Using cURL:
```bash
# Query the database
curl -X POST "http://localhost:8000/mcp/sql-query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top 5 customers"}'

# Search policies
curl -X POST "http://localhost:8000/mcp/policy-search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Return policy"}'
```

#### Using Python:
```python
import requests

# Chat with supervisor
response = requests.post(
    "http://localhost:8000/chat",
    json={"query": "What are my top customers?"}
)
print(response.json())

# Direct SQL query
response = requests.post(
    "http://localhost:8000/mcp/sql-query",
    json={"question": "Total sales this month?"}
)
print(response.json())
```

#### Using JavaScript/Frontend:
```javascript
// Fetch chat response
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    query: 'Show top customers',
    context: {}
  })
})
.then(res => res.json())
.then(data => console.log(data.response));

// Fetch sales summary
fetch('http://localhost:8000/mcp/sales-summary')
  .then(res => res.json())
  .then(data => console.log(data.result));
```

### Option 3: Update Streamlit to Call API Directly (Optional)
To decouple Streamlit from direct agent calls, modify `ui/app.py`:

```python
import requests

# Instead of: response = await asyncio.to_thread(run_agent, query)
# Use:
response = requests.post(
    "http://localhost:8000/chat",
    json={"query": query, "context": context}
)
result = response.json()
```

---

## Benefits

| Benefit | Reason |
|---------|--------|
| **Scalability** | Run multiple API instances behind a load balancer |
| **Separation of Concerns** | UI and backend are decoupled |
| **Reusability** | Any frontend (React, Vue, Mobile) can call the same API |
| **Tool Discovery** | `/mcp/tools-list` allows dynamic UI generation |
| **Monitoring** | FastAPI provides built-in logging, metrics, request tracing |
| **Testing** | Easier to unit test individual tools via HTTP |
| **Deployment** | Container-friendly, cloud-ready (AWS, Azure, GCP) |
| **Documentation** | Swagger UI auto-generated from code |

---

## Next Steps

1. ✅ Test endpoints at http://localhost:8000/docs
2. ✅ Try calling individual tools via cURL or Postman
3. ✅ Modify Streamlit to call the API (optional but recommended)
4. ✅ Add authentication (FastAPI Security)
5. ✅ Add rate limiting (Slowapi)
6. ✅ Deploy to production (Docker, Heroku, AWS Lambda, etc.)

---

## Production Deployment Tips

### Add Authentication
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, credentials = Depends(security)):
    # Validate token
    return ...
```

### Add Rate Limiting
```bash
pip install slowapi
```

### Docker Deployment
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
OPENAI_MODEL=gpt-4o
OPENAI_API_KEY=your-key
DATABASE_URL=sqlite:///data/shop_accounts.db
```

---

## Troubleshooting

### Issue: API won't start
**Solution:** Check if port 8000 is in use. Change it: `uvicorn api.main:app --port 8001`

### Issue: CORS errors from Streamlit
**Solution:** Already added CORS middleware. If issues persist, specify exact origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    ...
)
```

### Issue: Slow responses
**Solution:** Use `asyncio.to_thread()` for CPU-heavy tasks (already done). For I/O, consider async database drivers.

---

## Summary

You now have a **production-ready FastAPI backend** with:
- ✅ All MCP tools exposed as REST endpoints
- ✅ LangGraph multi-agent supervisor
- ✅ Streamlit frontend (optional)
- ✅ Automatic API documentation
- ✅ Error handling & CORS
- ✅ Ready for mobile/external client integration
