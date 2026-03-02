import logging
from graph.state import AgentState
from tools.file_tools import download_file, check_extension, is_extension_supported, extract_content
from db.checkpoint import save_error_checkpoint

logger = logging.getLogger(__name__)


def file_processor_node(state: AgentState) -> AgentState:
    """
    Agent 1 — File Processor
    Responsibilities:
      1. Download file bytes from file URL
      2. Detect file extension and whether it is an image
      3. Check if extension is supported
      4. Extract raw text content (skipped for images — handled by ocr_translation node)
    """
    run_id = state.get("run_id", "unknown")

    try:
        logger.info(f"[file_processor] Downloading file from: {state['file_url']}")
        file_bytes = download_file(state["file_url"])
    except Exception as e:
        error = f"Failed to download file: {e}"
        logger.error(f"[file_processor] {error}")
        save_error_checkpoint(run_id, "file_processor:download_file", state, error)
        return {**state, "error": error, "failed_node": "file_processor"}

    try:
        logger.info("[file_processor] Checking file extension")
        # check_extension now returns { "extension": str, "is_image": bool }
        ext_result = check_extension(file_bytes)
        extension = ext_result["extension"]
        is_image = ext_result["is_image"]
    except Exception as e:
        error = f"Failed to check extension: {e}"
        logger.error(f"[file_processor] {error}")
        save_error_checkpoint(run_id, "file_processor:check_extension", state, error)
        return {**state, "file_bytes": file_bytes, "error": error, "failed_node": "file_processor"}

    try:
        logger.info(f"[file_processor] Checking if extension is supported: {extension}")
        supported = is_extension_supported(extension)
    except Exception as e:
        error = f"Failed to check if extension is supported: {e}"
        logger.error(f"[file_processor] {error}")
        save_error_checkpoint(run_id, "file_processor:is_extension_supported", state, error)
        return {**state, "file_bytes": file_bytes, "extension": extension, "error": error, "failed_node": "file_processor"}

    if not supported:
        logger.warning(f"[file_processor] Unsupported file type: {extension}")
        return {**state, "file_bytes": file_bytes, "extension": extension, "is_supported": False, "is_image": is_image}

    # For image formats, skip extract_content — ocr_translation node will handle it
    if is_image:
        logger.info(f"[file_processor] Image format detected ({extension}) — skipping text extraction, routing to OCR")
        return {
            **state,
            "file_bytes": file_bytes,
            "extension": extension,
            "is_supported": True,
            "is_image": True,
            "raw_text": None,
            "error": None,
        }

    try:
        logger.info(f"[file_processor] Extracting content from {extension} file")
        raw_text = extract_content(file_bytes, extension)
    except Exception as e:
        error = f"Failed to extract content: {e}"
        logger.error(f"[file_processor] {error}")
        save_error_checkpoint(run_id, "file_processor:extract_content", state, error)
        return {**state, "file_bytes": file_bytes, "extension": extension, "is_supported": True, "is_image": False, "error": error, "failed_node": "file_processor"}

    logger.info("[file_processor] File processing complete")
    return {
        **state,
        "file_bytes": file_bytes,
        "extension": extension,
        "is_supported": True,
        "is_image": False,
        "raw_text": raw_text,
        "error": None,
    }