import logging
from graph.state import AgentState
from tools.normalizer_tools import normalize_urls
from db.checkpoint import save_error_checkpoint

logger = logging.getLogger(__name__)


def url_normalizer_node(state: AgentState) -> dict:
    """
    Agent 3 — URL Normalizer
    Returns ONLY the keys it changes to avoid parallel write conflicts.
    """
    run_id = state.get("run_id", "unknown")
    extracted_urls = state.get("extracted_urls", [])

    if not extracted_urls:
        logger.info("[url_normalizer] No URLs to normalize")
        return {"normalized_urls": []}

    try:
        logger.info(f"[url_normalizer] Normalizing {len(extracted_urls)} URLs")
        normalized = normalize_urls(extracted_urls)
        logger.info("[url_normalizer] URL normalization complete")
        return {"normalized_urls": normalized}

    except Exception as e:
        error = f"URL normalization failed: {e}"
        logger.error(f"[url_normalizer] {error}")
        save_error_checkpoint(run_id, "url_normalizer", state, error)
        return {"normalized_urls": extracted_urls, "error": error, "failed_node": "url_normalizer"}