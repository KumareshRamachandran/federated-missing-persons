"""
shared/tests/test_privacy.py

Unit tests for Privacy & Cryptography module:
  - Differential Privacy (dp_utils.py)
  - Partial SMPC & Homomorphic Encryption (smpc.py)
  - Secure Aggregation (secure_aggregation.py)
  - Privacy Evaluator (privacy_evaluator.py)
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from privacy.dp_utils import clip_gradients, add_gaussian_noise, compute_epsilon
from privacy.smpc import SMPCEncryptor, aggregate_encrypted_updates
from privacy.secure_aggregation import generate_mask_pair, apply_mask, simulate_secure_aggregation
from privacy.privacy_evaluator import PrivacyEvaluator


def test_clip_gradients():
    """Test PyTorch gradient clipping utility."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    x = torch.randn(4, 10)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    criterion = nn.MSELoss()
    
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    max_norm = 0.5
    total_norm = clip_gradients(model, max_norm=max_norm)
    assert total_norm >= 0.0

    # Verify clipped parameter gradient norm
    clipped_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            clipped_norm += p.grad.data.norm(2).item() ** 2
    clipped_norm = float(clipped_norm ** 0.5)
    assert clipped_norm <= max_norm + 1e-4


def test_add_gaussian_noise():
    """Test adding DP noise to model weights."""
    weights = [np.ones((5, 5), dtype=np.float32), np.zeros((10,), dtype=np.float32)]
    noise_mult = 1.0
    noisy_weights = add_gaussian_noise(weights, noise_multiplier=noise_mult, sensitivity=1.0)

    assert len(noisy_weights) == 2
    assert noisy_weights[0].shape == (5, 5)
    assert noisy_weights[1].shape == (10,)

    # Zero noise multiplier test
    clean_weights = add_gaussian_noise(weights, noise_multiplier=0.0)
    assert np.allclose(clean_weights[0], weights[0])


def test_compute_epsilon():
    """Test epsilon privacy budget calculation."""
    eps = compute_epsilon(
        noise_multiplier=1.1,
        num_samples=1000,
        batch_size=32,
        num_rounds=10,
        delta=1e-5
    )
    assert isinstance(eps, float)
    assert eps > 0.0
    assert eps != float('inf')


def test_smpc_encrypt_decrypt_aggregate():
    """Test SMPC / Homomorphic Encryption encryption, decryption, and aggregation."""
    encryptor = SMPCEncryptor()
    
    weights_client1 = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    weights_client2 = [np.array([4.0, 5.0, 6.0], dtype=np.float32)]

    enc1 = encryptor.encrypt_weights(weights_client1)
    enc2 = encryptor.encrypt_weights(weights_client2)

    # Coordinator aggregates without decrypting
    aggregated_enc = aggregate_encrypted_updates([enc1, enc2], context=encryptor.context)
    
    # Decrypt at node
    decrypted = encryptor.decrypt_weights(aggregated_enc, original_shapes=[(3,)])

    expected = weights_client1[0] + weights_client2[0]
    np.testing.assert_allclose(decrypted[0], expected, rtol=1e-2, atol=1e-2)


def test_secure_aggregation():
    """Test pairwise mask generation and cancellation in secure aggregation."""
    shape = (4, 4)
    mask_a, mask_b = generate_mask_pair(shape, seed=42)

    # Masks must cancel out exactly
    np.testing.assert_allclose(mask_a + mask_b, np.zeros(shape), atol=1e-6)

    weights_a = [np.ones(shape, dtype=np.float32)]
    weights_b = [np.full(shape, 2.0, dtype=np.float32)]

    masked_a = apply_mask(weights_a, [mask_a])
    masked_b = apply_mask(weights_b, [mask_b])

    aggregated = simulate_secure_aggregation([masked_a, masked_b])

    # Aggregated sum must equal unmasked sum (3.0)
    expected = weights_a[0] + weights_b[0]
    np.testing.assert_allclose(aggregated[0], expected, atol=1e-5)


def test_privacy_evaluator():
    """Test PrivacyEvaluator metrics and report generation."""
    evaluator = PrivacyEvaluator()

    for r in range(5):
        evaluator.record_round(
            epsilon=0.5 * (r + 1),
            acc_dp=0.92 - (0.01 * r),
            acc_no_dp=0.95
        )

    drop = evaluator.compute_accuracy_drop()
    assert drop > 0.0

    report = evaluator.generate_report()
    assert report["total_rounds_evaluated"] == 5
    assert report["final_epsilon"] == 2.5
    assert "accuracy_drop" in report

    # Test MIA simulation
    train_emb = np.random.randn(20, 512)
    test_emb = np.random.randn(20, 512)
    mia_auc = evaluator.membership_inference_attack(None, train_emb, test_emb)
    assert 0.0 <= mia_auc <= 1.0

    # Test gradient inversion resistance
    orig_emb = np.random.randn(512)
    noisy_grad = orig_emb + np.random.normal(0, 0.5, 512)
    reconstruction_err = evaluator.gradient_inversion_resistance(orig_emb, noisy_grad)
    assert reconstruction_err > 0.0
