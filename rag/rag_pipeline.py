"""
rag/rag_pipeline.py
-------------------
RAG pipeline using FAISS + OpenAI embeddings.

Two uses:
  1. Search T&C / policy documents (answer customer policy questions)
  2. Contextualise bill understanding (provide billing terminology + rules to LLM)

Functions:
  index_documents()      — build/rebuild FAISS index from DB T&C records
  clear_vector_store()   — delete index files (Streamlit "Clear Embeddings" button)
  search(query, k)       — retrieve top-k relevant chunks
  answer_rag_query()     — full RAG chain: retrieve → generate answer
  get_bill_context()     — retrieve billing/invoice related context for image analysis
"""

import os, shutil
from pathlib import Path
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
LEGACY_VECTOR_STORE_PATH = "./data/vector_store"


def _default_vector_store_root() -> Path:
    configured = os.getenv("VECTOR_STORE_PATH", LEGACY_VECTOR_STORE_PATH)
    if configured != LEGACY_VECTOR_STORE_PATH:
        return Path(configured)

    local_app_data = os.getenv("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "ShopOS" / "vector_store"
    return Path(LEGACY_VECTOR_STORE_PATH)


STORE_PATH = _default_vector_store_root() / "faiss_index"
LEGACY_STORE_PATH = Path(LEGACY_VECTOR_STORE_PATH) / "faiss_index"

# ── Models ───────────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              api_key=os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                 temperature=0.1,
                 api_key=os.getenv("OPENAI_API_KEY"))

# ── Text splitter ─────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=120,
    separators=["\n\n", "\n", ".", " "]
)

# ── Prompts ───────────────────────────────────────────────────────────────────
POLICY_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful shop assistant. Answer the customer's question using ONLY
the policy context provided. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Provide a clear, concise answer referencing specific policy sections where relevant.
""")

BILL_CONTEXT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert at reading shop invoices and bills.
Use the billing guidelines and policy context below to help analyse the bill image.

Billing Context:
{context}

Bill Analysis Task: {task}

Provide a structured analysis including all extracted fields and any policy-relevant observations.
""")

# ── Global store handle ───────────────────────────────────────────────────────
_store: FAISS | None = None


def _load_store() -> FAISS | None:
    """Load existing FAISS index from disk."""
    global _store
    if _store is not None:
        return _store
    candidate_paths = [STORE_PATH]
    if LEGACY_STORE_PATH != STORE_PATH:
        candidate_paths.append(LEGACY_STORE_PATH)

    for candidate in candidate_paths:
        if candidate.exists() and (candidate / "index.faiss").exists():
            _store = FAISS.load_local(
                str(candidate),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            break
    return _store


def index_documents() -> int:
    """
    Reads all active TermsConditions rows from DB, splits into chunks,
    embeds and saves a FAISS index.  Returns number of chunks indexed.
    """
    global _store
    from db.database import SessionLocal
    from db.models import TermsConditions

    with SessionLocal() as db:
        tcs = db.query(TermsConditions).filter(TermsConditions.active == True).all()

    if not tcs:
        return 0

    # Build LangChain Document objects
    raw_docs = [
        Document(
            page_content=f"TITLE: {t.title}\nCATEGORY: {t.category}\n\n{t.content}",
            metadata={"title": t.title, "category": t.category,
                      "version": t.version, "source": f"tc_{t.id}"},
        )
        for t in tcs
    ]

    chunks = splitter.split_documents(raw_docs)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    _store = FAISS.from_documents(chunks, embeddings)
    _store.save_local(str(STORE_PATH))
    return len(chunks)


def clear_vector_store() -> str:
    """
    Deletes the FAISS index from disk and resets in-memory store.
    Called by the Streamlit 'Clear Embeddings' button.
    """
    global _store
    _store = None
    cleared = False
    for candidate in [STORE_PATH, LEGACY_STORE_PATH]:
        if candidate.exists():
            shutil.rmtree(candidate)
            cleared = True
    if cleared:
        return "✅ Vector store cleared successfully."
    return "ℹ️ Vector store was already empty."


def is_indexed() -> bool:
    """Returns True if a FAISS index file exists on disk."""
    return any(
        (candidate / "index.faiss").exists()
        for candidate in [STORE_PATH, LEGACY_STORE_PATH]
        if candidate.exists()
    )


def search(query: str, k: int = 4) -> List[Document]:
    """Retrieve top-k relevant document chunks for a query."""
    store = _load_store()
    if store is None:
        raise RuntimeError("Vector store not initialised. Run index_documents() first.")
    return store.similarity_search(query, k=k)


def answer_rag_query(query: str, k: int = 4) -> Dict[str, Any]:
    """
    Full RAG chain for policy questions.
    Returns: { answer, sources, chunks_used, error }
    """
    from core.rag_handler import RAGHandler
    handler = RAGHandler()
    return handler.query(query, k)


def get_bill_context(task_description: str = "Extract all invoice fields") -> str:
    """
    Retrieves billing-related context chunks to augment bill image analysis.
    Specifically looks for: invoice structure, billing abbreviations, payment status.
    """
    try:
        # Search for billing-specific context
        billing_query = "invoice structure bill reading payment status GST abbreviations"
        docs = search(billing_query, k=3)
        context = "\n\n---\n".join(
            f"[{d.metadata.get('title','Doc')}]\n{d.page_content}" for d in docs
        )
        return context
    except Exception:
        # Fallback: return minimal context so bill analysis still works
        return "Standard invoice fields: invoice number, date, customer, items, subtotal, GST, total, payment status."
