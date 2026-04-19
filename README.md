# ShopOS - Accounts Management System

AI-powered shop management with Text-to-SQL, bill scanning, policy search, protected login, and memory-aware multi-agent chat.

## What This App Does

| Feature | Technology Used |
|---|---|
| Natural language to SQL queries | LangChain + OpenAI |
| Bill and invoice image analysis | Vision + RAG context |
| Policy and T&C search | FAISS vector store + embeddings |
| Conversational AI assistant | LangGraph multi-agent supervisor |
| Auth + protected sessions | FastAPI + Streamlit + SQLite sessions |
| Memory | Short-term session memory + long-term user memory |
| Reasoning pattern | Planning + reflection workflow |
| Tool execution | LangChain `@tool` decorators |
| Tracing and observability | LangSmith |
| Frontend | Streamlit |
| Backend API | FastAPI |
| Database | SQLite + SQLAlchemy |

## Project Structure

```text
ShopOs/
|-- agents/
|   `-- graph.py
|-- api/
|   `-- main.py
|-- core/
|-- db/
|   |-- database.py
|   |-- models.py
|   `-- seed.py
|-- guardrails/
|-- mcp/
|-- rag/
|-- services/
|   |-- auth.py
|   `-- memory.py
|-- ui/
|   `-- app.py
`-- data/
```

## Architecture

```text
User Input -> Guardrails -> Planner -> Supervisor -> Specialist Agent -> Tool Node -> Reflector -> Response
```

Specialist routes:

- `sql_agent` for customers, sales, invoices, products, balances
- `rag_agent` for policies, returns, warranties, payment terms
- `bill_agent` for uploaded bill and invoice files
- `analytics_agent` for summaries, overdue invoices, and stock alerts
- `general_agent` for broad chat and mixed requests

## Setup

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Configure environment

```bash
cp .env.example .env
```

3. Run the Streamlit app

```bash
streamlit run ui/app.py
```

4. Run the FastAPI server

```bash
uvicorn api.main:app --reload
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_MODEL` | No | Model name, default `gpt-4o` |
| `LANGCHAIN_API_KEY` | No | LangSmith key |
| `LANGCHAIN_TRACING_V2` | No | Enable tracing |
| `LANGCHAIN_PROJECT` | No | LangSmith project |
| `DATABASE_URL` | No | SQLite database URL |
| `SHOPOS_DATA_DIR` | No | Override runtime data directory |
| `VECTOR_STORE_PATH` | No | Vector store folder |

Default runtime database path: `./data/runtime/shop_accounts.db`

## New Additions

- FastAPI auth endpoints for signup, login, logout, and current-user lookup
- Login and signup flow in Streamlit
- Short-term memory saved per session
- Long-term memory saved per user when durable preferences are detected
- Planner and reflector stages wrapped around the existing agent flow
- Cleaner UI and API structure with less duplicated code
