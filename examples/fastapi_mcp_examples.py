"""
examples/fastapi_mcp_examples.py
---------------------------------
Practical examples of using MCP tools with FastAPI from different clients.
Run FastAPI first: uvicorn api.main:app --reload (port 8000)
Then try these examples.
"""

import requests
import json
import base64
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# API BASE URL
# ══════════════════════════════════════════════════════════════════════════════
API_BASE_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Chat with Supervisor Agent
# ══════════════════════════════════════════════════════════════════════════════
def example_supervisor_chat():
    """
    Send a natural language query to the supervisor agent.
    The supervisor decides which specialist agent to route to.
    """
    print("\n🤖 EXAMPLE 1: Supervisor Agent Chat")
    print("=" * 70)
    
    queries = [
        "Show me the top 5 customers by total sales",
        "What is the return policy?",
        "Show me all overdue invoices",
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query, "context": {}}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {data['response'][:100]}...")
            print(f"🧭 Route: {data.get('route', 'N/A')}")
            print(f"🔧 Tools Used: {', '.join(data.get('tools_used', []))}")
        else:
            print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Direct SQL Query Tool
# ══════════════════════════════════════════════════════════════════════════════
def example_sql_query():
    """
    Directly call the SQL query MCP tool to ask database questions.
    """
    print("\n📊 EXAMPLE 2: Direct SQL Query Tool")
    print("=" * 70)
    
    questions = [
        "What are the top 5 customers by total sales?",
        "Show me all invoices from last month",
        "Which products have inventory below 10 units?",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        
        response = requests.post(
            f"{API_BASE_URL}/mcp/sql-query",
            json={"question": question}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Result: {data['result'][:150]}...")
        else:
            print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Policy Search (RAG)
# ══════════════════════════════════════════════════════════════════════════════
def example_policy_search():
    """
    Search shop policies using RAG (Retrieval-Augmented Generation).
    """
    print("\n📚 EXAMPLE 3: Policy Search (RAG Tool)")
    print("=" * 70)
    
    queries = [
        "What is the return policy for electronics?",
        "How long do we give for payment?",
        "What warranty do we offer?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        
        response = requests.post(
            f"{API_BASE_URL}/mcp/policy-search",
            json={"query": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Result: {data['result'][:200]}...")
        else:
            print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Bill Analysis (Vision + RAG)
# ══════════════════════════════════════════════════════════════════════════════
def example_bill_analysis(image_path: str):
    """
    Analyze a bill/invoice image using GPT-4o Vision + RAG context.
    """
    print("\n📄 EXAMPLE 4: Bill Analysis (Vision + RAG)")
    print("=" * 70)
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   To test this, provide a valid bill/invoice image path.")
        return
    
    # Read and encode image to base64
    with open(image_path, "rb") as f:
        file_base64 = base64.b64encode(f.read()).decode()
    
    filename = Path(image_path).name
    print(f"\n📸 Analyzing: {filename}")
    
    response = requests.post(
        f"{API_BASE_URL}/mcp/analyze-bill",
        json={
            "file_base64": file_base64,
            "filename": filename
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Analysis: {data['result'][:200]}...")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Customer Summary
# ══════════════════════════════════════════════════════════════════════════════
def example_customer_summary():
    """
    Get full customer profile including contact, balance, and history.
    """
    print("\n👥 EXAMPLE 5: Customer Summary")
    print("=" * 70)
    
    customers = [
        "Akash Enterprises",
        "TechCorp",
        "Local Store",
    ]
    
    for customer in customers:
        print(f"\n🔎 Customer: {customer}")
        
        response = requests.post(
            f"{API_BASE_URL}/mcp/customer-summary",
            json={"customer_name": customer}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Profile: {data['result'][:150]}...")
        else:
            print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Sales Summary
# ══════════════════════════════════════════════════════════════════════════════
def example_sales_summary():
    """
    Get aggregated sales analytics and trends.
    """
    print("\n📈 EXAMPLE 6: Sales Summary")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/mcp/sales-summary")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Sales Data:\n{json.dumps(data, indent=2)[:300]}...")
    else:
        print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: Overdue Invoices
# ══════════════════════════════════════════════════════════════════════════════
def example_overdue_invoices():
    """
    Check all pending and overdue invoices.
    """
    print("\n⚠️  EXAMPLE 7: Overdue Invoices")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/mcp/overdue-invoices")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Overdue Invoices:\n{json.dumps(data, indent=2)[:300]}...")
    else:
        print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 8: Low Stock Alerts
# ══════════════════════════════════════════════════════════════════════════════
def example_low_stock_alerts():
    """
    Get list of products below reorder level.
    """
    print("\n📦 EXAMPLE 8: Low Stock Alerts")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/mcp/low-stock-alerts")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Low Stock Items:\n{json.dumps(data, indent=2)[:300]}...")
    else:
        print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 9: Discover Available Tools
# ══════════════════════════════════════════════════════════════════════════════
def example_discover_tools():
    """
    List all available MCP tools and their endpoints.
    """
    print("\n🔧 EXAMPLE 9: Discover Available Tools")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/mcp/tools-list")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ Available MCP Tools:")
        for tool in data['tools']:
            print(f"\n  📌 {tool['name']}")
            print(f"     Endpoint: {tool['endpoint']} ({tool['method']})")
            print(f"     Description: {tool['description']}")
    else:
        print(f"❌ Error: {response.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 MCP + FastAPI Integration Examples")
    print("=" * 70)
    print("\n⚠️  Make sure FastAPI is running: uvicorn api.main:app --reload")
    print(f"\n📍 API Base URL: {API_BASE_URL}")
    
    try:
        # Check if API is running
        response = requests.get(f"{API_BASE_URL}/")
        print(f"✅ API is running!\n")
        
        # Run examples
        example_discover_tools()
        example_supervisor_chat()
        example_sql_query()
        example_policy_search()
        example_customer_summary()
        example_sales_summary()
        example_overdue_invoices()
        example_low_stock_alerts()
        
        # Uncomment to test bill analysis (provide valid image path)
        # example_bill_analysis("path/to/bill.jpg")
        
        print("\n" + "=" * 70)
        print("✅ All examples completed!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to FastAPI server!")
        print(f"   Make sure it's running: uvicorn api.main:app --reload")
        print(f"   Expected URL: {API_BASE_URL}")
