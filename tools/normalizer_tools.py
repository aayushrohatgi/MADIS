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
                time.sleep(2 ** attempt)
    raise last_error


def normalize_urls(urls: list[str]) -> list[str]:
    """
    Normalizes a list of URLs:
    - Adds https:// if protocol is missing
    - Decodes any percent-encodings and re-encodes correctly
    """
    def _call():
        response = requests.post(
            f"{TOOLS_API_BASE_URL}/normalize/urls",
            json={"urls": urls},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["urls"]

    return _call_with_retry(_call)


def normalize_phones(phones: list[str]) -> list[str]:
    """
    Normalizes a list of phone numbers:
    - Removes spaces, dashes, brackets, and other formatting characters
    - Retains the country code prefix (already inferred by extraction agent)
    """
    def _call():
        response = requests.post(
            f"{TOOLS_API_BASE_URL}/normalize/phones",
            json={"phones": phones},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["phones"]

    return _call_with_retry(_call)