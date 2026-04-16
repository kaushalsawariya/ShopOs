"""
core/rag_handler.py
-------------------
Modular RAG query handler with token optimization.
"""

from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .token_manager import token_manager
from .retry_handler import retry_handler
from rag.rag_pipeline import search as rag_search


class RAGHandler:
    """Optimized RAG query handler with token management."""

    def __init__(self, max_context_tokens: int = 1500, max_output_tokens: int = 500):
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens

    def _optimize_documents(self, docs: List[Document], query: str) -> str:
        """Optimize retrieved documents for context."""
        if not docs:
            return "No relevant documents found."

        # Sort by relevance (assuming similarity_search returns in order)
        # Combine and optimize content
        context_parts = []
        total_tokens = 0

        for doc in docs:
            content = doc.page_content
            title = doc.metadata.get('title', 'Document')

            # Format document
            formatted = f"[{title}]\n{content}"
            doc_tokens = token_manager.count_tokens(formatted)

            # Check if adding this document would exceed limit
            if total_tokens + doc_tokens > self.max_context_tokens:
                # Truncate this document if possible
                remaining_tokens = self.max_context_tokens - total_tokens
                if remaining_tokens > 100:  # Minimum useful content
                    truncated = token_manager.truncate_text(formatted, remaining_tokens)
                    context_parts.append(truncated)
                break

            context_parts.append(formatted)
            total_tokens += doc_tokens

        return "\n\n---\n".join(context_parts)

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create optimized RAG prompt."""
        template = """Answer the question based on the following context. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

        return ChatPromptTemplate.from_template(template)

    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Execute optimized RAG query."""
        def _make_rag_call():
            # Retrieve documents
            docs = rag_search(question, k=k)

            # Optimize context
            context = self._optimize_documents(docs, question)

            # Check if we can make this request
            prompt_tokens = token_manager.count_tokens(context) + token_manager.count_tokens(question) + 100
            if not token_manager.rate_limiter.can_make_request(prompt_tokens + self.max_output_tokens):
                # Wait if needed instead of throwing exception
                token_manager.rate_limiter.wait_if_needed(prompt_tokens + self.max_output_tokens)
                # Check again after waiting
                if not token_manager.rate_limiter.can_make_request(prompt_tokens + self.max_output_tokens):
                    raise Exception("Rate limit would be exceeded even after waiting")

            # Create and execute chain
            from rag.rag_pipeline import llm

            chain = (
                {"context": lambda _: context, "question": RunnablePassthrough()}
                | self._create_prompt()
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke(question)

            # Record token usage
            answer_tokens = token_manager.count_tokens(answer)
            total_tokens = prompt_tokens + answer_tokens
            token_manager.rate_limiter.record_request(total_tokens)

            # Extract sources
            sources = list({doc.metadata.get("title", "Unknown") for doc in docs})

            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(docs),
                "error": None
            }

        try:
            return retry_handler.execute_with_retry(_make_rag_call)
        except Exception as e:
            return {
                "answer": "",
                "sources": [],
                "chunks_used": 0,
                "error": str(e)
            }