import logging
from graph.state import AgentState

logger = logging.getLogger(__name__)


def aggregator_node(state: AgentState) -> AgentState:
    """
    Aggregator — Final Node
    Merges results from the parallel URL and phone branches into the final output.
    """
    urls = state.get("normalized_urls") or []
    phones = state.get("normalized_phones") or []

    logger.info(f"[aggregator] Final result — URLs: {len(urls)}, Phone numbers: {len(phones)}")

    return {
        **state,
        "normalized_urls": urls,
        "normalized_phones": phones,
    }