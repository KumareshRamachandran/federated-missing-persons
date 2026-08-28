"""
shared/tests/test_pipeline.py

Unit and integration tests for Aswin Maheswaran's end-to-end Pipeline class.
Verifies:
  - Pipeline initialization and default configuration
  - Local Differential Privacy (LDP) noise injection
  - Inference matching across distributed nodes
  - Performance metrics and latency tracking

Author: Aswin Maheswaran (23BCE8540) — UI Dashboard & Integration
"""

import pytest
import numpy as np
import os
import tempfile
from PIL import Image

from dashboard.integration.pipeline import Pipeline


class TestIntegrationPipeline:
    @pytest.fixture
    def pipeline(self):
        return Pipeline()

    @pytest.fixture
    def sample_image(self):
        # Create a temporary dummy RGB face image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img = Image.new('RGB', (112, 112), color=(200, 150, 100))
            img.save(tmp, format='JPEG')
            path = tmp.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_pipeline_initialization(self, pipeline):
        assert pipeline.config is not None
        assert "nodes" in pipeline.config
        assert len(pipeline.latency_log) == 0

    def test_ldp_noise_preserves_shape_and_norm(self, pipeline):
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)

        noisy = pipeline.apply_ldp_noise(emb, noise_multiplier=0.05)
        assert noisy.shape == (512,)
        assert np.isclose(np.linalg.norm(noisy), 1.0, atol=1e-5)
        assert not np.array_equal(emb, noisy)

    def test_run_from_embedding(self, pipeline):
        emb = np.random.randn(512).astype(np.float32)
        res = pipeline.run_from_embedding(emb, apply_ldp=True)

        assert res["success"] is True
        assert res["embedding_protected"] is True
        assert res["nodes_queried"] >= 1
        assert "node_police" in res["results"]
        assert "node_hospital" in res["results"]
        assert "node_ngo" in res["results"]
        assert res["latency_ms"] >= 0.0

    def test_run_on_image(self, pipeline, sample_image):
        res = pipeline.run(sample_image, apply_ldp=True, noise_multiplier=0.05)
        assert res["success"] is True
        assert res["face_detected"] is True
        assert "results" in res
        assert len(pipeline.latency_log) >= 1

    def test_performance_metrics(self, pipeline):
        emb = np.random.randn(512).astype(np.float32)
        pipeline.run_from_embedding(emb)
        pipeline.run_from_embedding(emb)

        metrics = pipeline.get_performance_metrics()
        assert metrics["total_queries"] >= 2
        assert metrics["avg_latency_ms"] >= 0.0
        assert "rank1_accuracy_percentage" in metrics
