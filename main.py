import uuid
import json
import logging

from langgraph.checkpoint.postgres import PostgresSaver
from graph.graph import build_graph
from config import POSTGRES_CONN_STRING

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_agent(file_url: str) -> dict:
    """
    Entry point to run the extraction agent on a given file URL.

    Args:
        file_url: Public URL of the file to process.

    Returns:
        dict with keys: run_id, status, urls, phone_numbers, error (if any)
    """
    run_id = str(uuid.uuid4())
    logger.info(f"Starting agent run. run_id={run_id}, file_url={file_url}")

    initial_state = {
        "file_url": file_url,
        "run_id": run_id,
        "file_bytes": None,
        "extension": None,
        "is_supported": None,
        "raw_text": None,
        "extracted_urls": None,
        "extracted_phones": None,
        "normalized_urls": None,
        "normalized_phones": None,
        "error": None,
        "failed_node": None,
    }

    # PostgresSaver must be used as a context manager
    with PostgresSaver.from_conn_string(POSTGRES_CONN_STRING) as checkpointer:
        checkpointer.setup()  # creates LangGraph checkpoint tables if they don't exist
        agent_graph = build_graph(checkpointer)

        # thread_id ties all checkpoints for this run together in Postgres
        config = {"configurable": {"thread_id": run_id}}
        final_state = agent_graph.invoke(initial_state, config=config)

    # Build clean output
    if not final_state.get("is_supported"):
        result = {
            "run_id": run_id,
            "status": "unsupported_file_type",
            "extension": final_state.get("extension"),
            "urls": [],
            "phone_numbers": [],
        }
    elif final_state.get("error"):
        result = {
            "run_id": run_id,
            "status": "error",
            "failed_node": final_state.get("failed_node"),
            "error": final_state.get("error"),
            "urls": final_state.get("normalized_urls") or [],
            "phone_numbers": final_state.get("normalized_phones") or [],
        }
    else:
        result = {
            "run_id": run_id,
            "status": "success",
            "urls": final_state.get("normalized_urls") or [],
            "phone_numbers": final_state.get("normalized_phones") or [],
        }

    logger.info(f"Agent run complete. run_id={run_id}, status={result['status']}")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <file_url>")
        sys.exit(1)

    output = run_agent(sys.argv[1])
    print(json.dumps(output, indent=2))