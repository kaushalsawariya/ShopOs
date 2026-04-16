# ShopOS Review And Debug Notes

## What I fixed

- Preserved uploaded file names from the Streamlit UI into the agent pipeline so PDFs are correctly identified as PDFs.
- Changed the bill agent to process attached files directly instead of creating a one-off tool that the shared LangGraph tool executor could not run.
- Fixed tool routing so results return to the same agent that requested them instead of always flowing back through `sql_agent`.
- Made token counting resilient when `tiktoken` cannot download model encoding assets in restricted or offline environments.
- Added a PDF text fallback using `PyPDF2` so text-based PDFs can still be processed when `pdf2image` fails on Windows.

## Seed data improvements

- Added more RAG source material:
  - `GST and Tax Invoice Notes`
  - `Supplier Delivery and Receiving Policy`
- Updated seeding to backfill missing policy documents on later runs instead of only working on a brand-new database.
- Ensured demo data includes visible low-stock products.
- Ensured demo data includes a few overdue invoices for analytics and collections use cases.

## Why PDFs were failing

There were two main issues:

1. The uploaded filename was not reaching the bill analyzer, so PDF uploads could be treated like images.
2. The app depended on `pdf2image`, which usually requires Poppler on Windows. Without it, scanned PDFs fail during conversion.

The code now handles both better:

- filename-aware routing is fixed
- text-based PDFs can fall back to text extraction
- scanned PDFs still work best when Poppler is installed

## Remaining practical limitation

For scanned PDFs on Windows, install Poppler and make sure it is available on `PATH` if you want reliable page-to-image conversion for OCR-style analysis. Without that, the fallback only helps for PDFs that already contain extractable text.

## Suggested smoke checks

- `venv\Scripts\python -c "from agents.graph import run_agent; print('graph ok')"`
- `venv\Scripts\python -c "from db.seed import seed_database; seed_database(); print('seed ok')"`
- `venv\Scripts\python -c "from rag.rag_pipeline import is_indexed; print(is_indexed())"`
- `streamlit run ui/app.py`
