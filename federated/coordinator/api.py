"""
federated/coordinator/api.py

FastAPI REST API for the coordinator server.
Used by the Streamlit dashboard and external clients.

Endpoints:
  POST /search          — Submit a missing person photo for federated search
  GET  /status          — Federation round status
  GET  /accuracy        — Model accuracy history
  GET  /nodes           — List registered org nodes
  POST /nodes/register  — Register a new org node
  DELETE /nodes/{id}    — Deregister an org node
  GET  /stats           — Query statistics (no personal data)

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from federated.coordinator.model_manager import ModelManager
from federated.coordinator.query_router import QueryRouter

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Shared state (initialised on startup)
# ──────────────────────────────────────────────────────────────────

model_manager: Optional[ModelManager] = None
query_router: Optional[QueryRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup, clean up on shutdown."""
    global model_manager, query_router
    model_manager = ModelManager(model_dir="models/global")
    query_router = QueryRouter()
    logger.info("API server ready. ModelManager and QueryRouter initialised.")
    yield
    logger.info("API server shutting down.")


# ──────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Federated Missing Person Search — Coordinator API",
    description=(
        "Privacy-preserving federated search across distributed organization databases. "
        "No raw images or gallery data are transmitted — only binary match results."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────

class NodeRegistration(BaseModel):
    node_id: str
    address: str
    org_type: str = "unknown"   # e.g. "police", "hospital", "ngo"


class SearchResponse(BaseModel):
    query_id: str
    timestamp: str
    nodes_queried: int
    any_match: bool
    results: Dict
    latency_ms: float
    error: Optional[str] = None


class StatusResponse(BaseModel):
    current_round: int
    connected_nodes: int
    last_aggregation: Optional[str]
    best_accuracy: float


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(photo: UploadFile = File(...)):
    """
    Submit a missing person search query.

    Accepts a photo upload, generates a face embedding,
    broadcasts the embedding (NOT the photo) to all registered org nodes,
    and returns aggregated Match/No-Match results.

    Privacy: org gallery data is never transmitted — only binary results returned.
    """
    if query_router is None:
        raise HTTPException(status_code=503, detail="Query router not initialised.")

    if not query_router.list_nodes():
        raise HTTPException(
            status_code=503,
            detail="No org nodes registered. Start org node clients first."
        )

    # Save uploaded photo to a temp file
    suffix = os.path.splitext(photo.filename or "query.jpg")[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(photo.file, tmp)
        tmp_path = tmp.name

    try:
        result = query_router.handle_query(image_path=tmp_path)
    finally:
        os.unlink(tmp_path)  # Always delete temp file

    return JSONResponse(content=result)


@app.get("/status", response_model=StatusResponse, tags=["Federation"])
async def federation_status():
    """
    Get current federation round status.

    Returns the latest round number, connected node count,
    timestamp of last aggregation, and best accuracy achieved.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialised.")

    latest = model_manager.get_latest_record()
    return {
        "current_round": latest["round"] if latest else 0,
        "connected_nodes": len(query_router.list_nodes()) if query_router else 0,
        "last_aggregation": latest["timestamp"] if latest else None,
        "best_accuracy": model_manager.get_best_accuracy(),
    }


@app.get("/accuracy", tags=["Federation"])
async def accuracy_history(last_n: int = Query(default=50, ge=1, le=500)):
    """
    Get model accuracy history over all federation rounds.

    Args:
        last_n: Return only the last N rounds (default 50).

    Returns:
        {"history": [{version, round, accuracy, timestamp}]}
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialised.")

    history = model_manager.get_accuracy_history()
    return {"history": history[-last_n:]}


@app.get("/nodes", tags=["Nodes"])
async def list_nodes():
    """List all currently registered org nodes."""
    if query_router is None:
        raise HTTPException(status_code=503, detail="Query router not initialised.")
    nodes = query_router.list_nodes()
    return {"nodes": nodes, "count": len(nodes)}


@app.post("/nodes/register", tags=["Nodes"])
async def register_node(reg: NodeRegistration):
    """
    Register a new org node with the coordinator.

    Note: In the FL training loop, nodes connect directly via Flower gRPC.
    This endpoint is for the query routing layer (inference-time registration).
    """
    if query_router is None:
        raise HTTPException(status_code=503, detail="Query router not initialised.")

    # For inference-time queries, we need a proxy client
    # In production, this would connect to the node's gRPC inference endpoint
    # For simulation, we register a placeholder
    class RemoteNodeProxy:
        """Placeholder proxy for a remote org node."""
        def __init__(self, node_id: str, address: str):
            self.node_id = node_id
            self.address = address

        def query(self, embedding):
            # TODO: In production, make HTTP/gRPC call to node's local_matcher endpoint
            logger.warning("RemoteNodeProxy.query() called — implement real RPC for production.")
            return {"match": False, "confidence": 0.0, "note": "proxy_placeholder"}

    proxy = RemoteNodeProxy(reg.node_id, reg.address)
    query_router.register_node(reg.node_id, proxy)

    return {
        "message": f"Node '{reg.node_id}' registered.",
        "node_id": reg.node_id,
        "address": reg.address,
        "org_type": reg.org_type,
    }


@app.delete("/nodes/{node_id}", tags=["Nodes"])
async def deregister_node(node_id: str):
    """Remove an org node from the coordinator's query registry."""
    if query_router is None:
        raise HTTPException(status_code=503, detail="Query router not initialised.")

    success = query_router.deregister_node(node_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    return {"message": f"Node '{node_id}' deregistered."}


@app.get("/stats", tags=["Analytics"])
async def query_statistics():
    """
    Return aggregate search statistics.
    No personal data is stored — only counts and latencies.
    """
    if query_router is None:
        raise HTTPException(status_code=503, detail="Query router not initialised.")
    return query_router.get_query_stats()


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_manager": model_manager is not None,
        "query_router": query_router is not None,
        "nodes_online": len(query_router.list_nodes()) if query_router else 0,
    }


# ──────────────────────────────────────────────────────────────────
# Run directly (for development)
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "federated.coordinator.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
