import json
import uuid
import logging
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import Json

from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS error_checkpoints (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      TEXT NOT NULL,
    failed_node TEXT NOT NULL,
    state       JSONB NOT NULL,
    error       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def ensure_table_exists():
    """Creates the error_checkpoints table if it doesn't exist."""
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()


def save_error_checkpoint(run_id: str, failed_node: str, state: dict, error: str):
    """
    Persists a failed agent state to Postgres so it can be replayed or forked later.

    Args:
        run_id:      Unique identifier for this agent run.
        failed_node: Name of the node that failed (e.g. 'file_processor').
        state:       The full AgentState dict at the point of failure.
        error:       The error message or exception string.
    """
    try:
        ensure_table_exists()

        # bytes are not JSON-serializable — convert to None for storage
        serializable_state = {
            k: (None if isinstance(v, bytes) else v)
            for k, v in state.items()
        }

        with _get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO error_checkpoints (run_id, failed_node, state, error)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (run_id, failed_node, Json(serializable_state), error),
                )
            conn.commit()

        logger.info(f"Error checkpoint saved for run_id={run_id}, node={failed_node}")

    except Exception as db_err:
        # Never let checkpoint saving crash the main error handler
        logger.error(f"Failed to save error checkpoint: {db_err}")


def load_checkpoint(run_id: str) -> list[dict]:
    """
    Loads all checkpoints for a given run_id, ordered by creation time.
    Useful for replay or forking flows later.
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_id, failed_node, state, error, created_at
                FROM error_checkpoints
                WHERE run_id = %s
                ORDER BY created_at ASC
                """,
                (run_id,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": str(row[0]),
            "run_id": row[1],
            "failed_node": row[2],
            "state": row[3],
            "error": row[4],
            "created_at": row[5].isoformat(),
        }
        for row in rows
    ]