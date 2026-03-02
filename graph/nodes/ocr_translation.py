import base64
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState
from db.checkpoint import save_error_checkpoint
from config import GEMINI_API_KEY, GEMINI_OCR_MODEL

logger = logging.getLogger(__name__)

# TODO: Refine this prompt when discussing prompts
SYSTEM_PROMPT = """You are an OCR and translation specialist.
Your task is to:
1. Extract all text visible in the provided image (OCR)
2. If the extracted text is not in English, translate it to English
3. If the text is already in English, return it as-is without any modification

Return ONLY the extracted (and translated if necessary) plain text.
No explanations, no formatting, no markdown — just the raw text content.
"""


def ocr_translation_node(state: AgentState) -> dict:
    """
    Node 1.5 — OCR + Translation
    Uses Gemini to:
    - Extract text from image bytes (OCR)
    - Translate to English if not already in English
    Output is raw_text, which feeds into the existing url/phone extractor nodes.
    """
    run_id = state.get("run_id", "unknown")

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_OCR_MODEL,
            google_api_key=GEMINI_API_KEY,
        )

        # Encode image bytes to base64 for Gemini vision input
        image_bytes = state["file_bytes"]
        extension = state.get("extension", "jpeg").lstrip(".")
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Map extension to valid MIME type
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(extension.lower(), "image/jpeg")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": "Extract and translate (if needed) all text from this image.",
                },
            ]),
        ]

        logger.info(f"[ocr_translation] Running OCR + translation on {extension} image using {GEMINI_OCR_MODEL}")
        response = llm.invoke(messages)

        raw_text = _extract_text(response)
        logger.info(f"[ocr_translation]: {raw_text}")
        logger.info(f"[ocr_translation] OCR complete — extracted {len(raw_text)} characters")

        return {"raw_text": raw_text}

    except Exception as e:
        error = f"OCR + translation failed: {e}"
        logger.error(f"[ocr_translation] {error}")
        save_error_checkpoint(run_id, "ocr_translation", state, error)
        return {"raw_text": None, "error": error, "failed_node": "ocr_translation"}


def _extract_text(response) -> str:
    """Safely extracts text from a LangChain response regardless of content format."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"].strip()
            if isinstance(block, str):
                return block.strip()
    return ""