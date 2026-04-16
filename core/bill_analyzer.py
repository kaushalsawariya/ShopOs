"""
core/bill_analyzer.py
---------------------
Modular bill/invoice analysis component with token optimization.
Supports both images and PDFs.
"""

import json
import base64
import pathlib
import io
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pdf2image import convert_from_bytes
from PIL import Image

from .token_manager import token_manager
from .retry_handler import retry_handler
from rag.rag_pipeline import get_bill_context
from guardrails.guardrails import sanitize_output


class BillAnalyzer:
    """Modular bill analysis with token optimization."""

    def __init__(self, model: str = "gpt-4o", max_tokens: int = 1000):
        self.model = model
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
        )

    def _get_optimized_context(self, task_description: str = "Extract all invoice fields") -> str:
        """Get optimized billing context from RAG."""
        try:
            bill_context = get_bill_context(task_description)
            # Optimize context to fit within token budget
            max_context_tokens = 500  # Reserve tokens for the actual analysis
            return token_manager.optimize_context(bill_context, max_context_tokens)
        except Exception:
            return "Standard invoice fields: invoice number, date, customer, items, subtotal, GST, total, payment status."

    def _build_extraction_prompt(self, context: str) -> str:
        """Build an optimized extraction prompt."""
        return f"""Extract invoice data from the image. Use this context: {context}

Return ONLY a JSON object with these fields:
{{
  "invoice_number": "string or null",
  "vendor_name": "string or null",
  "customer_name": "string or null",
  "date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "subtotal": 0,
  "cgst": 0,
  "sgst": 0,
  "total_amount": 0,
  "payment_method": "string or null",
  "payment_status": "paid/pending/partial/overdue or null",
  "currency": "INR"
}}"""

    def _convert_pdf_to_images(self, pdf_bytes: bytes) -> List[str]:
        """Convert PDF pages to base64 encoded images."""
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt='JPEG', size=(None, 1200))

            base64_images = []
            for i, img in enumerate(images):
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=95)
                img_bytes = img_buffer.getvalue()

                # Convert to base64
                base64_str = base64.b64encode(img_bytes).decode('utf-8')
                base64_images.append(base64_str)

                # Limit to first 5 pages to avoid excessive processing
                if i >= 4:
                    break

            return base64_images
        except Exception as e:
            raise Exception(f"PDF conversion failed: {str(e)}")

    def _extract_data_from_image(self, image_base64: str, filename: str) -> Dict[str, Any]:
        """Extract structured data from a single image."""
        # Get optimized context
        context = self._get_optimized_context()
        prompt = self._build_extraction_prompt(context)

        # Determine media type
        ext = pathlib.Path(filename).suffix.lower()
        media_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"
        }.get(ext.lstrip("."), "image/jpeg")

        def _make_vision_call():
            # Check if we can make this request
            estimated_tokens = token_manager.count_tokens(prompt) + 170  # Image tokens estimate
            if not token_manager.rate_limiter.can_make_request(estimated_tokens):
                # Wait if needed instead of throwing exception
                token_manager.rate_limiter.wait_if_needed(estimated_tokens)
                # Check again after waiting
                if not token_manager.rate_limiter.can_make_request(estimated_tokens):
                    raise Exception("Rate limit would be exceeded even after waiting")

            message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}",
                        "detail": "high"
                    }
                },
                {"type": "text", "text": prompt}
            ])

            response = self.llm.invoke([message])
            raw_content = response.content.strip()
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()

            # Record token usage
            tokens_used = token_manager.count_tokens(raw_content)
            token_manager.rate_limiter.record_request(tokens_used)

            return json.loads(raw_content)

        try:
            return retry_handler.execute_with_retry(_make_vision_call)
        except json.JSONDecodeError:
            return {"error": "Could not parse JSON from image"}
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}

    def _extract_data(self, file_base64: str, filename: str) -> Dict[str, Any]:
        """Extract structured data from bill (image or PDF)."""
        ext = pathlib.Path(filename).suffix.lower()

        if ext == ".pdf":
            # Handle PDF files
            try:
                pdf_bytes = base64.b64decode(file_base64)
                images_base64 = self._convert_pdf_to_images(pdf_bytes)

                if not images_base64:
                    return {"error": "PDF contains no readable pages"}

                # Extract from first page (or combine results from multiple pages)
                results = []
                for i, img_b64 in enumerate(images_base64):
                    result = self._extract_data_from_image(img_b64, f"{filename}_page_{i+1}.jpg")
                    results.append(result)

                    # If we have valid data from first page, use it
                    if "error" not in result and result.get("total_amount", 0) > 0:
                        return result

                # If no valid data found, return the first result or combined error
                if results:
                    return results[0]
                else:
                    return {"error": "Could not extract data from PDF"}

            except Exception as e:
                return {"error": f"PDF processing failed: {str(e)}"}
        else:
            # Handle image files
            return self._extract_data_from_image(file_base64, filename)

    def _generate_analysis(self, extracted_data: Dict[str, Any]) -> str:
        """Generate human-readable analysis."""
        if "error" in extracted_data:
            return f"Could not analyze bill: {extracted_data['error']}"

        prompt = f"""Summarize this invoice in 2-3 sentences:
{json.dumps(extracted_data, indent=2)}

Cover: what the bill is for, key amounts, and payment status. Use ₹ for currency."""

        try:
            # Check token limits
            prompt_tokens = token_manager.count_tokens(prompt)
            if not token_manager.rate_limiter.can_make_request(prompt_tokens + 200):
                # Wait if needed instead of skipping
                token_manager.rate_limiter.wait_if_needed(prompt_tokens + 200)
                # Check again after waiting
                if not token_manager.rate_limiter.can_make_request(prompt_tokens + 200):
                    return "Analysis skipped due to persistent rate limits."

            response = self.llm.invoke([HumanMessage(content=prompt)])

            # Record usage
            tokens_used = token_manager.count_tokens(response.content)
            token_manager.rate_limiter.record_request(tokens_used)

            return response.content.strip()

        except Exception as e:
            return f"Analysis generation failed: {str(e)}"

    def _save_bill_file(self, file_base64: str, filename: str):
        """Save bill file (image or PDF) for record keeping."""
        try:
            pathlib.Path("./data/bills").mkdir(parents=True, exist_ok=True)
            file_bytes = base64.b64decode(file_base64)
            with open(f"./data/bills/{filename}", "wb") as f:
                f.write(file_bytes)
        except Exception:
            pass  # Non-critical operation

    def analyze_bill(self, file_base64: str, filename: str = "bill.jpg") -> str:
        """Complete bill analysis pipeline."""
        # Extract data
        extracted = self._extract_data(file_base64, filename)

        # Generate analysis
        analysis = self._generate_analysis(extracted)

        # Save file
        self._save_bill_file(file_base64, filename)

        # Format response
        result = f"EXTRACTED DATA:\n{json.dumps(extracted, indent=2)}\n\nANALYSIS:\n{analysis}"

        return sanitize_output(result)