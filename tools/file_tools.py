import time
import requests
from config import TOOLS_API_BASE_URL, MAX_RETRIES


def _call_with_retry(fn, *args, **kwargs):
    """Calls fn up to MAX_RETRIES+1 times with exponential backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
    raise last_error


def download_file(file_url: str) -> bytes:
    """Downloads the file from a public URL and returns raw bytes."""
    def _call():
        response = requests.post(
            f"{TOOLS_API_BASE_URL}/files/download",
            json={"url": file_url},
            timeout=30,
        )
        response.raise_for_status()
        return response.content

    return _call_with_retry(_call)


def check_extension(file_bytes: bytes) -> dict:
    """
    Detects the file extension from raw bytes (magic bytes / content sniffing).

    Returns:
        {
            "extension": str,   e.g. "pdf", "jpg", "docx"
            "is_image": bool    True for image formats (jpg, png, gif, webp, etc.)
        }
    """
    def _call():
        response = requests.post(
            f"{TOOLS_API_BASE_URL}/files/check-extension",
            files={"file": file_bytes},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()  # expects { "extension": str, "is_image": bool }

    return _call_with_retry(_call)


def is_extension_supported(extension: str) -> bool:
    """Returns True if the file extension is supported for content extraction."""
    def _call():
        response = requests.get(
            f"{TOOLS_API_BASE_URL}/files/is-supported",
            params={"extension": extension},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["supported"]

    return _call_with_retry(_call)


def extract_content(file_bytes: bytes, extension: str) -> str:
    """
    Extracts raw text content from the file bytes based on its extension.
    Only called for non-image formats — images are handled by the ocr_translation node.
    """
    def _call():
        response = requests.post(
            f"{TOOLS_API_BASE_URL}/files/extract-content",
            files={"file": file_bytes},
            data={"extension": extension},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["text"]

    return _call_with_retry(_call)