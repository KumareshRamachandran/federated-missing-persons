"""
privacy/privacy_evaluator.py

Evaluates the privacy-utility tradeoff of the system.
Measures:
  - Privacy budget consumed (epsilon, delta) per round
  - Accuracy drop due to DP noise (with vs without DP)
  - Resistance to membership inference attacks (basic simulation)
  - Gradient inversion attack resistance

Member: K Kishore (23BCE9746) — Cryptography & Privacy
"""

import numpy as np


class PrivacyEvaluator:
    """Evaluates and documents the privacy-utility tradeoff."""

    def __init__(self):
        self.epsilon_history = []    # Privacy budget per round
        self.accuracy_with_dp = []
        self.accuracy_without_dp = []

    def record_round(self, epsilon: float, acc_dp: float, acc_no_dp: float):
        """Record metrics for one federation round."""
        self.epsilon_history.append(epsilon)
        self.accuracy_with_dp.append(acc_dp)
        self.accuracy_without_dp.append(acc_no_dp)

    def compute_accuracy_drop(self) -> float:
        """
        Compute average accuracy drop caused by Differential Privacy.

        Returns:
            avg_drop: Average percentage drop in accuracy across all recorded rounds.
        """
        if not self.accuracy_with_dp or not self.accuracy_without_dp:
            return 0.0

        drops = [no_dp - dp for dp, no_dp in zip(self.accuracy_with_dp, self.accuracy_without_dp)]
        return float(np.mean(drops))

    def membership_inference_attack(self, model, train_embeddings: list, test_embeddings: list) -> float:
        """
        Simulate a basic membership inference attack (MIA).
        Checks if a shadow model / threshold classifier can distinguish training (member)
        vs non-training (non-member) data from similarity / confidence scores.

        Returns:
            attack_auc: Closer to 0.50 = better privacy protection (random guessing).
        """
        train_arr = np.asarray(train_embeddings, dtype=np.float32) if len(train_embeddings) > 0 else np.empty((0, 512))
        test_arr = np.asarray(test_embeddings, dtype=np.float32) if len(test_embeddings) > 0 else np.empty((0, 512))

        if len(train_arr) == 0 or len(test_arr) == 0:
            return 0.50

        # Compute sample norms / feature variance as attack features
        train_scores = np.linalg.norm(train_arr, axis=-1)
        test_scores = np.linalg.norm(test_arr, axis=-1)

        labels = np.concatenate([np.ones(len(train_scores)), np.zeros(len(test_scores))])
        scores = np.concatenate([train_scores, test_scores])

        try:
            from sklearn.metrics import roc_auc_score
            from sklearn.linear_model import LogisticRegression

            X = scores.reshape(-1, 1)
            clf = LogisticRegression()
            clf.fit(X, labels)
            preds = clf.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(labels, preds))
            # MIA AUC should be close to 0.50 under strong DP
            return auc
        except Exception:
            # Threshold attack fallback
            threshold = np.median(scores)
            preds = (scores >= threshold).astype(int)
            acc = np.mean(preds == labels)
            return float(acc)

    def gradient_inversion_resistance(self, original_embedding: np.ndarray,
                                       noisy_gradient: np.ndarray) -> float:
        """
        Measure how much DP noise prevents gradient reconstruction attacks.

        Args:
            original_embedding: Ground truth embedding vector.
            noisy_gradient: Gradient update observed by adversary under DP noise.

        Returns:
            reconstruction_error: L2 norm distance (higher = stronger privacy protection).
        """
        orig = np.asarray(original_embedding, dtype=np.float32).flatten()
        grad = np.asarray(noisy_gradient, dtype=np.float32).flatten()

        # Trim or pad to match dimensions
        min_len = min(len(orig), len(grad))
        if min_len == 0:
            return 0.0

        orig_sub = orig[:min_len]
        grad_sub = grad[:min_len]

        reconstruction_error = float(np.linalg.norm(orig_sub - grad_sub))
        return reconstruction_error

    def generate_report(self) -> dict:
        """
        Generate a full privacy evaluation report summarizing DP budget and attack metrics.

        Returns:
            report_dict: Summary containing epsilon history, accuracy impact, and security scores.
        """
        final_epsilon = float(self.epsilon_history[-1]) if self.epsilon_history else 0.0
        avg_acc_dp = float(np.mean(self.accuracy_with_dp)) if self.accuracy_with_dp else 0.0
        avg_acc_no_dp = float(np.mean(self.accuracy_without_dp)) if self.accuracy_without_dp else 0.0
        acc_drop = self.compute_accuracy_drop()

        return {
            "total_rounds_evaluated": len(self.epsilon_history),
            "final_epsilon": round(final_epsilon, 4),
            "avg_accuracy_with_dp": round(avg_acc_dp, 4),
            "avg_accuracy_without_dp": round(avg_acc_no_dp, 4),
            "accuracy_drop": round(acc_drop, 4),
            "epsilon_trajectory": [round(e, 4) for e in self.epsilon_history]
        }

