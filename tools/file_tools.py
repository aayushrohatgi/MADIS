import base64
import asyncio
import logging
from mcp import ClientSession
from mcp.client.sse import sse_client
from config import MCP_SERVER_URL, MAX_RETRIES
import time

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Runs an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _call_with_retry(fn, *args, **kwargs):
    """Calls fn up to MAX_RETRIES+1 times with exponential backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise last_error


async def _call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Opens an SSE connection to the MCP server and calls the specified tool."""
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            # MCP returns content as a list of blocks — extract the first text block
            for block in result.content:
                if hasattr(block, "text"):
                    import json
                    return json.loads(block.text)
            raise RuntimeError(f"No text content returned from MCP tool: {tool_name}")


# ─── Public tool functions ────────────────────────────────────────────────────

def download_file(s3_url: str) -> bytes:
    """
    Downloads the file from a public S3 URL via the MCP server.
    Returns raw file bytes (decoded from Base64).
    """
    def _call():
        result = _run_async(_call_mcp_tool("download_file", {"s3Url": s3_url}))
        return base64.b64decode(result["content"])

    return _call_with_retry(_call)


def check_extension(file_bytes: bytes) -> dict:
    """
    Detects the file extension and image flag via the MCP server.
    Returns: { "extension": str, "is_image": bool }
    """
    def _call():
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        return _run_async(_call_mcp_tool("check_extension", {"base64FileBytes": b64}))

    return _call_with_retry(_call)


def is_extension_supported(extension: str) -> bool:
    """Returns True if the file extension is supported for content extraction."""
    def _call():
        result = _run_async(_call_mcp_tool("is_extension_supported", {"extension": extension}))
        return result["supported"]

    return _call_with_retry(_call)


def extract_content(file_bytes: bytes, extension: str) -> str:
    """
    Extracts raw text content from the file bytes via the MCP server.
    Only called for non-image formats.
    """
    def _call():
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        result = _run_async(_call_mcp_tool("extract_content", {"base64FileBytes": b64, "extension": extension}))
        return result["text"]

    return _call_with_retry(_call)