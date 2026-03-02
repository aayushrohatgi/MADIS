from langgraph.graph import StateGraph, END, START

from graph.nodes.aggregator import aggregator_node
from graph.nodes.file_processor import file_processor_node
from graph.nodes.ocr_translation import ocr_translation_node
from graph.nodes.phone_extractor import phone_extractor_node
from graph.nodes.phone_normalizer import phone_normalizer_node
from graph.nodes.url_extractor import url_extractor_node
from graph.nodes.url_normalizer import url_normalizer_node
from graph.state import AgentState


def route_after_file_processor(state: AgentState) -> list[str] | str:
    """
    Three-way conditional edge after Agent 1:
    - error or unsupported file type → END
    - image format                   → ocr_translation
    - non-image supported file       → parallel fan-out to url_extractor + phone_extractor
    """
    if state.get("error"):
        return END
    if not state.get("is_supported"):
        return END
    if state.get("is_image"):
        return "ocr_translation"
    # Non-image supported file — parallel fan-out
    return ["url_extractor", "phone_extractor"]


def route_after_ocr(state: AgentState) -> list[str] | str:
    """
    Edge after ocr_translation:
    - error → END
    - success → parallel fan-out to url_extractor + phone_extractor
    """
    if state.get("error"):
        return END
    return ["url_extractor", "phone_extractor"]


def build_graph(checkpointer) -> StateGraph:
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("file_processor", file_processor_node)
    graph.add_node("ocr_translation", ocr_translation_node)
    graph.add_node("url_extractor", url_extractor_node)
    graph.add_node("url_normalizer", url_normalizer_node)
    graph.add_node("phone_extractor", phone_extractor_node)
    graph.add_node("phone_normalizer", phone_normalizer_node)
    graph.add_node("aggregator", aggregator_node)

    # Entry point
    graph.add_edge(START, "file_processor")

    # Three-way conditional after file processing
    graph.add_conditional_edges(
        "file_processor",
        route_after_file_processor,
        {
            "ocr_translation": "ocr_translation",
            "url_extractor": "url_extractor",
            "phone_extractor": "phone_extractor",
            END: END,
        },
    )

    # After OCR — fan-out to extractors (or END on error)
    graph.add_conditional_edges(
        "ocr_translation",
        route_after_ocr,
        {
            "url_extractor": "url_extractor",
            "phone_extractor": "phone_extractor",
            END: END,
        },
    )

    # URL branch: extractor → normalizer → aggregator
    graph.add_edge("url_extractor", "url_normalizer")
    graph.add_edge("url_normalizer", "aggregator")

    # Phone branch: extractor → normalizer → aggregator
    graph.add_edge("phone_extractor", "phone_normalizer")
    graph.add_edge("phone_normalizer", "aggregator")

    # Aggregator → END
    graph.add_edge("aggregator", END)

    return graph.compile(checkpointer=checkpointer)