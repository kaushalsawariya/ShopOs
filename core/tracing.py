"""
core/tracing.py
Initialises LangSmith tracing. Called once at app startup.
All LangChain / LangGraph calls are auto-traced when env vars are set.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def setup_langsmith():
    """Configure LangSmith environment variables and print status."""
    key     = os.getenv("LANGCHAIN_API_KEY", "")
    enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "shop-accounts-capstone")

    if enabled and key and not key.startswith("ls__your"):
        print(f"✅ LangSmith tracing ON  →  project: {project}")
        print("   View at https://smith.langchain.com/")
    else:
        # Disable gracefully if no real key
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("ℹ️  LangSmith tracing OFF  (set LANGCHAIN_API_KEY to enable)")
