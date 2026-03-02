import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import AgentState
from db.checkpoint import save_error_checkpoint
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# TODO: Refine this prompt when discussing prompts
SYSTEM_PROMPT = """You are a URL extraction specialist.
Your task is to extract all URLs from the given text.
Return ONLY a JSON array of URL strings. No explanation, no markdown, no extra text.
Example output: ["http://example.com", "www.another.com/path"]
If no URLs are found, return an empty array: []
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


def url_extractor_node(state: AgentState) -> dict:
    """
    Agent 2 — URL Extractor
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
            HumanMessage(content=f"Extract all URLs from the following text:\n\n{state['raw_text']}"),
        ]

        logger.info("[url_extractor] Calling Gemini to extract URLs")
        response = llm.invoke(messages)

        text = _extract_text(response)
        extracted_urls = json.loads(text)
        logger.info(f"[url_extractor] Extracted {len(extracted_urls)} URLs")

        return {"extracted_urls": extracted_urls}

    except Exception as e:
        error = f"URL extraction failed: {e}"
        logger.error(f"[url_extractor] {error}")
        save_error_checkpoint(run_id, "url_extractor", state, error)
        return {"extracted_urls": [], "error": error, "failed_node": "url_extractor"}