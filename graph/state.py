from typing import Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Run identity
    run_id: str

    # Input
    file_url: str

    # File processing (Agent 1)
    file_bytes: Optional[bytes]
    extension: Optional[str]
    is_supported: Optional[bool]
    is_image: Optional[bool]
    raw_text: Optional[str]

    # Extraction (Agents 2 & 4)
    extracted_urls: Optional[list[str]]
    extracted_phones: Optional[list[str]]

    # Normalization (Agents 3 & 5)
    normalized_urls: Optional[list[str]]
    normalized_phones: Optional[list[str]]

    # Error handling
    error: Optional[str]
    failed_node: Optional[str]