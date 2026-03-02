import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState
from db.checkpoint import save_error_checkpoint
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# TODO: Refine this prompt when discussing prompts
SYSTEM_PROMPT = """You are a phone number extraction specialist.
Your task is to extract all phone numbers from the given text.

Important:
- Infer the country code from context clues in the text (addresses, country names, domain extensions, etc.)
- If you cannot infer the country, use +1 (US) as default
- Include the country code prefix in every number (e.g. +91, +1, +44)
- Return numbers in the format: +<country_code><number> (e.g. +911234567890)
- Return ONLY a JSON array of phone number strings. No explanation, no markdown, no extra text.

Example output: ["+911234567890", "+14155551234"]
If no phone numbers are found, return an empty array: []
"""


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


def phone_extractor_node(state: AgentState) -> dict:
    """
    Agent 4 — Phone Number Extractor
    Returns ONLY the keys it changes to avoid parallel write conflicts.
    """
    run_id = state.get("run_id", "unknown")

    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Extract all phone numbers from the following text:\n\n{state['raw_text']}"),
        ]

        logger.info("[phone_extractor] Calling Gemini to extract phone numbers")
        response = llm.invoke(messages)

        text = _extract_text(response)
        extracted_phones = json.loads(text)
        logger.info(f"[phone_extractor] Extracted {len(extracted_phones)} phone numbers")

        return {"extracted_phones": extracted_phones}

    except Exception as e:
        error = f"Phone extraction failed: {e}"
        logger.error(f"[phone_extractor] {error}")
        save_error_checkpoint(run_id, "phone_extractor", state, error)
        return {"extracted_phones": [], "error": error, "failed_node": "phone_extractor"}