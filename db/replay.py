import uuid
import logging
from langgraph.checkpoint.postgres import PostgresSaver
from config import POSTGRES_CONN_STRING

logger = logging.getLogger(__name__)


def get_failed_runs() -> list[dict]:
    """
    Returns all runs that have an error checkpoint saved in the error_checkpoints table.
    """
    import psycopg2
    from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

    with psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT run_id, failed_node, error, created_at
                FROM error_checkpoints
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()

    return [
        {
            "run_id": row[0],
            "failed_node": row[1],
            "error": row[2],
            "created_at": row[3].isoformat(),
        }
        for row in rows
    ]


def load_run_state(run_id: str) -> dict | None:
    """
    Loads the most recent checkpointed state for a given run_id.
    """
    with PostgresSaver.from_conn_string(POSTGRES_CONN_STRING) as checkpointer:
        config = {"configurable": {"thread_id": run_id}}
        checkpoint_tuple = checkpointer.get_tuple(config)

        if not checkpoint_tuple:
            logger.warning(f"[replay] No checkpoint found for run_id={run_id}")
            return None

        logger.info(f"[replay] Loaded checkpoint for run_id={run_id}")
        return checkpoint_tuple.checkpoint.get("channel_values", {})


def replay_from_checkpoint(run_id: str) -> dict:
    """
    Resumes a failed run on the SAME thread from its last checkpoint.
    LangGraph automatically skips already-completed nodes.

    Use for: transient failures (network errors, API timeouts) where
    the state is correct but execution failed.
    """
    from graph.graph import build_graph

    logger.info(f"[replay] Replaying run_id={run_id}")

    with PostgresSaver.from_conn_string(POSTGRES_CONN_STRING) as checkpointer:
        agent_graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": run_id}}
        final_state = agent_graph.invoke(None, config=config)

    return _build_output(run_id, final_state)


def fork_from_checkpoint(
    run_id: str,
    new_file_url: str | None = None,
    new_raw_text: str | None = None,
    state_overrides: dict | None = None,
) -> dict:
    """
    Forks a failed run into a NEW thread with optional changes applied.
    The original run is completely untouched.

    Forking behavior depends on what you change:

    - new_file_url:  Replaces the file_url and resets ALL downstream state.
                     The fork re-runs from file_processor (full re-download + re-extract).

    - new_raw_text:  Replaces the extracted text and resets extraction/normalization state.
                     The fork re-runs from url_extractor + phone_extractor (skips file_processor).

    - state_overrides: Any additional state keys to override (e.g. extracted_phones).
                       Applied last, on top of the above changes.

    Args:
        run_id:          The original run_id to fork from.
        new_file_url:    New file_url to use (triggers full re-run from file_processor).
        new_raw_text:    Corrected raw text (triggers re-run from extractors).
        state_overrides: Any additional state key overrides.

    Returns:
        Final output dict with new run_id, urls, and phone_numbers.
    """
    from graph.graph import build_graph

    base_state = load_run_state(run_id)
    if not base_state:
        raise ValueError(f"No checkpoint found for run_id={run_id}. Cannot fork.")

    forked_state = {**base_state}

    if new_file_url:
        logger.info(f"[replay] file_url override detected — resetting all downstream state")
        # Reset everything downstream of file_processor
        forked_state["file_url"] = new_file_url
        forked_state["file_bytes"] = None
        forked_state["extension"] = None
        forked_state["is_supported"] = None
        forked_state["raw_text"] = None
        forked_state["extracted_urls"] = None
        forked_state["extracted_phones"] = None
        forked_state["normalized_urls"] = None
        forked_state["normalized_phones"] = None

    elif new_raw_text:
        logger.info(f"[replay] raw_text override detected — resetting extraction/normalization state")
        # Keep file_processor results, reset everything downstream of it
        forked_state["raw_text"] = new_raw_text
        forked_state["extracted_urls"] = None
        forked_state["extracted_phones"] = None
        forked_state["normalized_urls"] = None
        forked_state["normalized_phones"] = None

    # Apply any additional overrides last
    if state_overrides:
        forked_state.update(state_overrides)

    # Clear error so the graph doesn't route to END immediately
    forked_state["error"] = None
    forked_state["failed_node"] = None

    # New run_id for independent tracking
    forked_run_id = str(uuid.uuid4())
    forked_state["run_id"] = forked_run_id

    logger.info(f"[replay] Fork: {run_id} → {forked_run_id}")

    with PostgresSaver.from_conn_string(POSTGRES_CONN_STRING) as checkpointer:
        checkpointer.setup()
        agent_graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": forked_run_id}}
        final_state = agent_graph.invoke(forked_state, config=config)

    return _build_output(forked_run_id, final_state)


def _build_output(run_id: str, final_state: dict) -> dict:
    """Builds a clean output dict from the final graph state."""
    if not final_state.get("is_supported"):
        return {
            "run_id": run_id,
            "status": "unsupported_file_type",
            "extension": final_state.get("extension"),
            "urls": [],
            "phone_numbers": [],
        }
    elif final_state.get("error"):
        return {
            "run_id": run_id,
            "status": "error",
            "failed_node": final_state.get("failed_node"),
            "error": final_state.get("error"),
            "urls": final_state.get("normalized_urls") or [],
            "phone_numbers": final_state.get("normalized_phones") or [],
        }
    else:
        return {
            "run_id": run_id,
            "status": "success",
            "urls": final_state.get("normalized_urls") or [],
            "phone_numbers": final_state.get("normalized_phones") or [],
        }