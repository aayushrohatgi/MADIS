import logging
from graph.state import AgentState
from tools.normalizer_tools import normalize_phones
from db.checkpoint import save_error_checkpoint

logger = logging.getLogger(__name__)


def phone_normalizer_node(state: AgentState) -> dict:
    """
    Agent 5 — Phone Number Normalizer
    Returns ONLY the keys it changes to avoid parallel write conflicts.
    """
    run_id = state.get("run_id", "unknown")
    extracted_phones = state.get("extracted_phones", [])

    if not extracted_phones:
        logger.info("[phone_normalizer] No phone numbers to normalize")
        return {"normalized_phones": []}

    try:
        logger.info(f"[phone_normalizer] Normalizing {len(extracted_phones)} phone numbers")
        normalized = normalize_phones(extracted_phones)
        logger.info("[phone_normalizer] Phone normalization complete")
        return {"normalized_phones": normalized}

    except Exception as e:
        error = f"Phone normalization failed: {e}"
        logger.error(f"[phone_normalizer] {error}")
        save_error_checkpoint(run_id, "phone_normalizer", state, error)
        return {"normalized_phones": extracted_phones, "error": error, "failed_node": "phone_normalizer"}