"""
Examples of how to use the replay and fork functionality.
Usage:
    python replay_example.py list
    python replay_example.py inspect        <run_id>
    python replay_example.py replay         <run_id>
    python replay_example.py fork-url       <run_id> <new_file_url>
    python replay_example.py fork-text      <run_id> <new_raw_text>
    python replay_example.py fork-override  <run_id>
"""
import sys
import json
import logging
from db.replay import get_failed_runs, load_run_state, replay_from_checkpoint, fork_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def example_list_failures():
    """List all failed runs from error_checkpoints table."""
    print("\n=== Failed Runs ===")
    failed = get_failed_runs()
    if not failed:
        print("No failed runs found.")
        return
    for run in failed:
        print(f"  run_id     : {run['run_id']}")
        print(f"  failed_node: {run['failed_node']}")
        print(f"  error      : {run['error']}")
        print(f"  created_at : {run['created_at']}")
        print()


def example_inspect_state(run_id: str):
    """Inspect the full state saved at the point of failure for a given run."""
    print(f"\n=== State at Failure for run_id={run_id} ===")
    state = load_run_state(run_id)
    if not state:
        print("No state found.")
        return
    for k, v in state.items():
        if k != "file_bytes":
            print(f"  {k}: {v}")


def example_replay(run_id: str):
    """
    Replay on the SAME thread with NO state changes.
    LangGraph resumes from the last successful checkpoint — completed nodes are skipped.

    Best for: transient failures (network errors, API timeouts)
    where the state is correct but execution just failed mid-way.
    """
    print(f"\n=== Replaying run_id={run_id} ===")
    print("Resuming from last checkpoint on the same thread...\n")

    result = replay_from_checkpoint(run_id)

    print("\n--- Result ---")
    print(json.dumps(result, indent=2))


def example_fork_with_new_url(run_id: str, new_file_url: str):
    """
    Fork with a NEW file URL.
    Resets ALL downstream state — the fork re-runs the full workflow from
    file_processor (re-download, re-extract, re-normalize everything).

    Best for: the original file was wrong or replaced with a corrected version.
    """
    print(f"\n=== Fork with new file URL ===")
    print(f"  Original run_id : {run_id}")
    print(f"  New file URL      : {new_file_url}\n")

    result = fork_from_checkpoint(
        run_id=run_id,
        new_file_url=new_file_url,
    )

    print("\n--- Result ---")
    print(json.dumps(result, indent=2))
    print(f"\nOriginal run_id : {run_id}")
    print(f"Forked   run_id : {result['run_id']}")


def example_fork_with_new_raw_text(run_id: str, new_raw_text: str):
    """
    Fork with corrected RAW TEXT.
    Keeps file_processor results (file_bytes, extension, is_supported) but resets
    all extraction and normalization state. The fork re-runs from url_extractor
    and phone_extractor onwards — file_processor is skipped.

    Best for: the file was downloaded correctly but content extraction
    produced garbled or incomplete text.
    """
    print(f"\n=== Fork with corrected raw text ===")
    print(f"  Original run_id : {run_id}")
    print(f"  New raw_text    : {new_raw_text[:80]}{'...' if len(new_raw_text) > 80 else ''}\n")

    result = fork_from_checkpoint(
        run_id=run_id,
        new_raw_text=new_raw_text,
    )

    print("\n--- Result ---")
    print(json.dumps(result, indent=2))
    print(f"\nOriginal run_id : {run_id}")
    print(f"Forked   run_id : {result['run_id']}")


def example_fork_with_state_overrides(run_id: str):
    """
    Fork with manual state overrides (e.g. inject corrected extracted data).
    Useful when extraction ran fine but produced wrong results and you want
    to manually correct and re-run only normalization.

    Best for: fine-grained corrections at any point in the pipeline.
    """
    print(f"\n=== Fork with manual state overrides ===")
    print(f"  Original run_id : {run_id}\n")

    result = fork_from_checkpoint(
        run_id=run_id,
        state_overrides={
            "extracted_phones": ["+911234567890", "+14155559876"],
            "extracted_urls": ["https://example.com", "https://another.com"],
        }
    )

    print("\n--- Result ---")
    print(json.dumps(result, indent=2))
    print(f"\nOriginal run_id : {run_id}")
    print(f"Forked   run_id : {result['run_id']}")


if __name__ == "__main__":
    commands = ["list", "inspect", "replay", "fork-url", "fork-text", "fork-override"]

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        example_list_failures()

    elif command == "inspect":
        run_id = sys.argv[2] if len(sys.argv) > 2 else None
        if not run_id:
            print("Error: inspect requires <run_id>"); sys.exit(1)
        example_inspect_state(run_id)

    elif command == "replay":
        run_id = sys.argv[2] if len(sys.argv) > 2 else None
        if not run_id:
            print("Error: replay requires <run_id>"); sys.exit(1)
        example_replay(run_id)

    elif command == "fork-url":
        if len(sys.argv) < 4:
            print("Error: fork-url requires <run_id> <new_file_url>"); sys.exit(1)
        example_fork_with_new_url(run_id=sys.argv[2], new_file_url=sys.argv[3])

    elif command == "fork-text":
        if len(sys.argv) < 4:
            print("Error: fork-text requires <run_id> <new_raw_text>"); sys.exit(1)
        example_fork_with_new_raw_text(run_id=sys.argv[2], new_raw_text=sys.argv[3])

    elif command == "fork-override":
        run_id = sys.argv[2] if len(sys.argv) > 2 else None
        if not run_id:
            print("Error: fork-override requires <run_id>"); sys.exit(1)
        example_fork_with_state_overrides(run_id)