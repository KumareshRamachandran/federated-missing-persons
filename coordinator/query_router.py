"""
coordinator/query_router.py

Handles search queries from investigators.
- Accepts a query image path
- Generates face embedding via face_engine
- Sends embedding to all registered org nodes
- Collects and aggregates Match/No-Match responses
"""

from face_engine.embedder import generate_embedding
from face_engine.detector import detect_face


def handle_query(image_path: str, node_clients: list) -> dict:
    """
    Process a missing person search query.

    Args:
        image_path: Path to the uploaded photo of the missing person.
        node_clients: List of connected org node stubs.

    Returns:
        dict: {node_id: {"match": bool, "confidence": float, "location": str}}
    """
    # TODO: Detect and align face from image_path
    # TODO: Generate 512-d ArcFace embedding
    # TODO: Broadcast embedding to each node_client
    # TODO: Collect binary responses
    # TODO: Return aggregated results dict
    pass
