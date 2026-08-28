"""
shared/tests/test_evaluation_metrics.py

Unit tests for biometric evaluation metrics and benchmark utilities.
Tests:
  - Rank-1 identification accuracy (closed-set & open-set)
  - Cumulative Match Characteristic (CMC)
  - False Match Rate (FMR) and False Non-Match Rate (FNMR)
  - Receiver Operating Characteristic (ROC) & Equal Error Rate (EER)
  - Benchmark comparison harness

Author: Aswin Maheswaran (23BCE8540) — UI Dashboard & Integration
"""

import pytest
import numpy as np

from dashboard.evaluation.metrics import (
    rank1_identification_rate,
    cumulative_match_characteristic,
    false_match_rate,
    false_non_match_rate,
    compute_roc,
    compute_eer,
)
from dashboard.evaluation.benchmark import (
    run_centralized_benchmark,
    run_federated_benchmark,
    compare_and_report,
)


class TestBiometricMetrics:
    @pytest.fixture
    def mock_embeddings(self):
        np.random.seed(42)
        dim = 512
        num_identities = 10

        gallery = {}
        queries = {}
        prototypes = {}

        for i in range(num_identities):
            pid = f"person_{i:03d}"
            proto = np.random.randn(dim).astype(np.float32)
            proto /= np.linalg.norm(proto)
            prototypes[pid] = proto

            # Gallery embedding (slight variation)
            g_emb = proto + np.random.normal(0, 0.01, dim).astype(np.float32)
            g_emb /= np.linalg.norm(g_emb)
            gallery[pid] = g_emb

            # Query embedding (slight variation)
            q_emb = proto + np.random.normal(0, 0.01, dim).astype(np.float32)
            q_emb /= np.linalg.norm(q_emb)
            queries[pid] = q_emb

        return queries, gallery, prototypes

    def test_rank1_identification_rate_perfect_match(self, mock_embeddings):
        queries, gallery, _ = mock_embeddings
        rate = rank1_identification_rate(queries, gallery, threshold=0.45)
        assert rate > 0.90, f"Expected high Rank-1 rate, got {rate}"

    def test_rank1_empty_inputs(self):
        assert rank1_identification_rate({}, {}) == 0.0

    def test_cmc_curve(self, mock_embeddings):
        queries, gallery, _ = mock_embeddings
        cmc = cumulative_match_characteristic(queries, gallery, ranks=[1, 3, 5])
        assert 1 in cmc and 3 in cmc and 5 in cmc
        assert cmc[1] <= cmc[3] <= cmc[5]

    def test_fmr_fnmr_calculations(self):
        # Genuine pairs have high similarity ~0.8-0.9
        genuine_scores = [0.85, 0.88, 0.92, 0.78, 0.81]
        # Impostor pairs have low similarity ~0.1-0.3
        impostor_scores = [0.15, 0.22, 0.18, 0.29, 0.12]

        # At threshold 0.5:
        # FMR (impostors >= 0.5) should be 0.0
        # FNMR (genuine < 0.5) should be 0.0
        fmr = false_match_rate(impostor_scores, threshold=0.50)
        fnmr = false_non_match_rate(genuine_scores, threshold=0.50)

        assert fmr == 0.0
        assert fnmr == 0.0

    def test_compute_roc_and_eer(self):
        genuine_scores = [0.9, 0.85, 0.8, 0.75, 0.7]
        impostor_scores = [0.1, 0.15, 0.2, 0.25, 0.3]

        roc = compute_roc(genuine_scores, impostor_scores)
        assert "thresholds" in roc
        assert "fmr" in roc
        assert "fnmr" in roc
        assert "eer" in roc
        assert "auc" in roc
        assert 0.0 <= roc["eer"] <= 1.0
        assert roc["auc"] >= 0.90


class TestBenchmarkEngine:
    def test_centralized_benchmark_execution(self):
        res = run_centralized_benchmark(num_synthetic=10)
        assert "rank1_accuracy" in res
        assert "avg_latency_ms" in res
        assert res["raw_biometric_exposure"] is True

    def test_federated_benchmark_execution(self):
        res = run_federated_benchmark(apply_ldp=True, num_synthetic=10)
        assert "rank1_accuracy" in res
        assert "avg_latency_ms" in res
        assert res["raw_biometric_exposure"] is False
        assert res["inference_privacy"] is True

    def test_compare_and_report(self):
        c = run_centralized_benchmark(num_synthetic=10)
        f = run_federated_benchmark(apply_ldp=True, num_synthetic=10)
        report = compare_and_report(c, f)
        assert "summary" in report
        assert "accuracy_drop_pct" in report["summary"]
