"""
federated/coordinator/query_router.py

Search Query Router — the coordinator's inference engine.

When an investigator uploads a photo of a missing person:
  1. Generate a 512-d ArcFace face embedding from the query photo
  2. Broadcast ONLY the embedding (not the photo) to all org nodes
  3. Each node runs local matching → returns Match/No-Match
  4. Aggregate results and return to the API layer

PRIVACY GUARANTEE:
  - Org nodes never receive the original photo, only a mathematical vector.
  - Nodes return only {match: bool, confidence: float} — no gallery data exposed.
  - The coordinator never sees org gallery contents.

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Manages search query broadcasting to all registered org nodes.

    Maintains a registry of active OrgFLClient instances
    (or lightweight proxy objects when running distributed over gRPC).
    """

    def __init__(self, nodes: Optional[List] = None, max_workers: int = 8):
        """
        Args:
            nodes:       List of OrgFLClient or compatible objects with a .query(embedding) method.
            max_workers: Thread pool size for parallel node querying.
        """
        self.nodes: Dict[str, object] = {}   # {node_id: client_obj}
        self.max_workers = max_workers
        self.query_log: List[dict] = []

        if nodes:
            for node in nodes:
                self.register_node(node.node_id, node)

    # ──────────────────────────────────────────────────────────────
    # Node Registry
    # ──────────────────────────────────────────────────────────────

    def register_node(self, node_id: str, client) -> None:
        """Register an org node client."""
        self.nodes[node_id] = client
        logger.info("Node registered: %s | total_nodes=%d", node_id, len(self.nodes))

    def deregister_node(self, node_id: str) -> bool:
        """Remove an org node from the registry."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info("Node deregistered: %s", node_id)
            return True
        return False

    def list_nodes(self) -> List[str]:
        """Return list of currently registered node IDs."""
        return list(self.nodes.keys())

    # ──────────────────────────────────────────────────────────────
    # Query Handling
    # ──────────────────────────────────────────────────────────────

    def handle_query(
        self,
        image_path: str,
        embedder_fn=None,
    ) -> Dict:
        """
        Full query pipeline: image → embedding → broadcast → aggregate results.

        Args:
            image_path:  Path to the investigator's uploaded photo.
            embedder_fn: Callable (image_path -> np.ndarray).
                         Defaults to vision.embedder.generate_embedding.

        Returns:
            {
              "query_id":   str,
              "timestamp":  str (ISO 8601),
              "nodes_queried": int,
              "any_match":  bool,
              "results": {
                "node_police":   {"match": bool, "confidence": float},
                "node_hospital": {"match": bool, "confidence": float},
                "node_ngo":      {"match": bool, "confidence": float},
              },
              "latency_ms": float,
            }
        """
        query_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        logger.info("[%s] Query started | image=%s | nodes=%d", query_id, image_path, len(self.nodes))

        # ── Step 1: Generate face embedding ───────────────────────
        embedding = self._generate_embedding(image_path, embedder_fn)
        if embedding is None:
            logger.warning("[%s] No face detected in query image.", query_id)
            return self._error_response(query_id, "No face detected in the uploaded image.")

        # ── Step 2: Broadcast to all nodes (parallel) ──────────────
        results = self._broadcast_query(query_id, embedding)

        # ── Step 3: Aggregate ──────────────────────────────────────
        any_match = any(r.get("match", False) for r in results.values())
        latency_ms = round((time.time() - start_time) * 1000, 2)

        response = {
            "query_id": query_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "nodes_queried": len(results),
            "any_match": any_match,
            "results": results,
            "latency_ms": latency_ms,
        }

        # Log query (for analytics, no personal data stored)
        self.query_log.append({
            "query_id": query_id,
            "timestamp": response["timestamp"],
            "nodes_queried": len(results),
            "any_match": any_match,
            "latency_ms": latency_ms,
        })

        logger.info(
            "[%s] Query complete | match=%s | latency=%.1fms",
            query_id, any_match, latency_ms,
        )
        return response

    def handle_query_from_embedding(self, embedding: np.ndarray) -> Dict:
        """
        Query directly from a pre-computed embedding (used by API for efficiency).

        Args:
            embedding: 512-d L2-normalised face embedding.

        Returns:
            Same structure as handle_query().
        """
        query_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        results = self._broadcast_query(query_id, embedding)
        any_match = any(r.get("match", False) for r in results.values())
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query_id": query_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "nodes_queried": len(results),
            "any_match": any_match,
            "results": results,
            "latency_ms": latency_ms,
        }

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _generate_embedding(
        self, image_path: str, embedder_fn=None
    ) -> Optional[np.ndarray]:
        """Generate a 512-d face embedding from an image path."""
        if embedder_fn is None:
            try:
                from vision.embedder import generate_embedding
                embedder_fn = generate_embedding
            except ImportError:
                logger.error("vision.embedder not available. Cannot generate embedding.")
                return None

        embedding = embedder_fn(image_path)
        if embedding is None:
            return None

        # Ensure L2 normalisation
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def _broadcast_query(
        self, query_id: str, embedding: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Send the query embedding to all registered nodes in parallel.
        Each node returns only {"match": bool, "confidence": float}.

        Uses ThreadPoolExecutor for concurrent node queries.
        """
        if not self.nodes:
            logger.warning("[%s] No nodes registered. Cannot broadcast query.", query_id)
            return {}

        results: Dict[str, Dict] = {}

        def query_node(node_id: str, client) -> tuple:
            try:
                result = client.query(embedding)
                logger.debug(
                    "[%s] Node %s → match=%s conf=%.4f",
                    query_id, node_id, result.get("match"), result.get("confidence"),
                )
                return node_id, result
            except Exception as e:
                logger.error("[%s] Node %s failed: %s", query_id, node_id, e)
                return node_id, {"match": False, "confidence": 0.0, "error": str(e)}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(query_node, nid, client): nid
                for nid, client in self.nodes.items()
            }
            for future in as_completed(futures):
                node_id, result = future.result()
                results[node_id] = result

        return results

    def _error_response(self, query_id: str, message: str) -> Dict:
        return {
            "query_id": query_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "nodes_queried": 0,
            "any_match": False,
            "results": {},
            "error": message,
            "latency_ms": 0.0,
        }

    def get_query_stats(self) -> Dict:
        """Return summary statistics from the query log."""
        if not self.query_log:
            return {"total_queries": 0}
        total = len(self.query_log)
        matches = sum(1 for q in self.query_log if q["any_match"])
        avg_latency = sum(q["latency_ms"] for q in self.query_log) / total
        return {
            "total_queries": total,
            "total_matches": matches,
            "match_rate": round(matches / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }
