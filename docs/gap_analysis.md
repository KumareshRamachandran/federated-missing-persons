# Literature Gap Analysis & Novelty Positioning
## Project: Privacy-Preserving Federated Learning for Missing Person Identification

---

## Critical Finding: Reference [1] is Our Own Prior Work

> **[1] Rakshika, Waikhom, Muthaiah — "AI-based Missing Person Identification using YOLO and Deep Facial Embeddings," ICCMC 2026**

This paper is the **centralized baseline** that our project directly extends. It:
- Achieved 97.50% accuracy using YOLO + ArcFace (centralized, no privacy)
- Used our exact custom dataset (780 images: 240 positive, 540 negative)
- Explicitly leaves the gap: *"Extend to real-time tracking using DeepSORT"* and has **no privacy whatsoever**

**This means our project is a well-motivated, direct federated + privacy extension of a published result from our own group. This is the strongest possible novelty justification.**

---

## Paper-by-Paper Gap Map

| Ref | What It Does | What It Lacks vs. Our Project |
|-----|-------------|-------------------------------|
| [1] | YOLO + ArcFace for missing person ID — centralized, our own prior work | No federation, no privacy, no cross-org deployment |
| [2] | FL + DP + XOR-masked feature quantization for face templates | Not domain-specific (missing persons); no YOLO surveillance pipeline; no real-time tracking |
| [3] | Survey of privacy techniques (HE, DP, SMPC, cancelable biometrics) | Survey only — no implementation, no missing person application |
| [4] | Vertical FL + Split Learning + P-SMPC + LDP for face recognition | Not missing-person specific; no surveillance distortions; no YOLO detection stage |
| [5] | Federated micro-expression recognition with RSA-HE | Different modality (micro-expressions, not identity matching) |
| [6] | Comparative FL study (FedAvg vs FedProx) on Non-IID face data | No privacy beyond DP; no application context; no real-world scenario |
| [7] | Blockchain + FL for face recognition (anti-poisoning) | No missing person use-case; no surveillance pipeline; centralized inference |
| [8] | Federated emotion recognition for advertising analytics | Entirely different application domain |
| [9] | FL for industrial face recognition + ESD integration | Industrial domain; no surveillance or identity search workflow |
| [10] | FL face recognition + GAN imposters + SMC | No missing person workflow; no YOLO; no surveillance distortions |
| [11] | Edge FL + DP + HE for video sensing (no missing persons) | No domain-specific application; no dashboard; no cross-org query |
| [12] | Blockchain + FL for recruitment face recognition | Different domain; low accuracy (65%), no privacy-preserving inference |
| [13] | Committee consensus blockchain FL aggregation | General FL aggregation — no biometric/missing person context |
| [14] | Few-shot FL for facial expression recognition | Different modality; no identity matching |
| [15] | Face de-identification against deep CNNs | Opposite goal — hiding faces, not identifying them |
| [16] | Augmentation for CNN face recognition | Augmentation study only; no FL, no privacy |

---

## The Research Gap Our Project Fills

After reviewing all 16 papers, **no existing work combines all of the following**:

1. **Domain:** Missing person identification across distributed organizations
2. **Detection:** Surveillance-adapted pipeline (YOLO → MTCNN → ArcFace), already validated in [1]
3. **Federation:** Privacy-preserving training AND inference across heterogeneous org nodes
4. **Inference-time privacy:** Only Match/No-Match returned — organizations never expose gallery data
5. **Combined privacy stack:** DP (Opacus) + SMPC/HE (TenSEAL) + Secure Aggregation together
6. **Federated feedback loop:** Model improves from confirmed real-world matches (not just training rounds)
7. **Surveillance conditions:** Dataset includes low-light, occlusion, compression artifacts

---

## Our 5 Novelty Claims (for the Paper)

### Novelty 1 — Federated Extension of a Published Baseline
We extend our own prior published work [1] (centralized, 97.50% accuracy) into a federated, privacy-preserving architecture. This gives us a direct apples-to-apples comparison: **centralized accuracy vs. federated accuracy with privacy guarantees**.

> *"While [1] demonstrated 97.50% accuracy in a centralized setting, it exposed all organizational databases to a central server. This work addresses that fundamental limitation."*

### Novelty 2 — Inference-Time Privacy (Not Just Training-Time)
Most FL papers ([2], [4], [6], [7], [10]) protect privacy during **training** but still centralize inference.  
We ensure that during a search query, **no gallery data ever leaves an organization** — only a binary Match/No-Match result is returned.

> *This is the key architectural contribution. No surveyed paper applies this to the missing person domain.*

### Novelty 3 — Surveillance-Realistic Federated Dataset
Unlike papers using clean CelebA or LFW, we use our custom dataset ([1]) simulating real CCTV conditions (low-light, blur, occlusion) and partition it federally across simulated Police, Hospital, and NGO nodes in a Non-IID split.

### Novelty 4 — Combined DP + SMPC (Training and Weight Protection)
- [2] uses DP + XOR masking (no HE)
- [4] uses P-SMPC alone
- [7] uses blockchain (no DP)
- We combine **Opacus DP-SGD** (gradient protection) + **TenSEAL CKKS HE** (weight encryption) — the most comprehensive privacy stack in any surveyed missing-person paper

### Novelty 5 — Federated Feedback Loop (Closed-Loop Learning)
No surveyed paper implements a feedback mechanism where **confirmed real-world matches trigger a local fine-tune and federated update**. This creates a continuously improving system — directly addressing [1]'s static centralized model.

---

## What We Must NOT Replicate

To avoid being accused of reproducing existing work:

| ❌ Do NOT do | ✅ Do THIS instead |
|------------|------------------|
| Centralized matching (like [1]) | Federated local matching only |
| DP alone as the privacy layer (like [2], [6]) | DP + SMPC/HE combination |
| Standard FedAvg without modification | FedAvg with DP noise injection + accuracy-weighted client selection |
| Static model (trained once, deployed) | Federated feedback loop: confirmed matches → fine-tune → aggregate |
| Clean academic datasets only | Custom surveillance dataset ([1]) + CelebA + LFW |
| Face recognition only (no detection) | Full pipeline: YOLO → MTCNN → ArcFace |

---

## Gaps We Can Also Address (Bonus Novelty)

From the explicit "Research Gap" column of our literature survey:

| Paper | Their Gap | Our Opportunity |
|-------|-----------|----------------|
| [1] | "Extend to real-time tracking with DeepSORT" | We can add DeepSORT to track persons across frames before querying FL |
| [2] | "Explore blockchain or SMPC for verifiable aggregation auditability" | We implement SMPC (TenSEAL) — directly addresses this gap |
| [4] | "Test on larger diverse datasets; adaptive DP" | We use larger CelebA + custom CCTV; can implement adaptive DP noise |
| [6] | "Optimize FL hyperparameters for Non-IID under strict DP" | Our Non-IID split + combined DP/SMPC directly addresses this |
| [11] | "Build real-time monitoring/alert dashboards" | We build a Streamlit dashboard — directly addresses this gap |

---

## Summary for Paper Abstract

> We present a Privacy-Preserving Federated Learning system for missing person identification that extends our prior centralized work [1] into a decentralized, multi-organizational framework. Unlike existing federated face recognition systems that focus solely on training-time privacy, our system introduces inference-time privacy guarantees — organizations return only binary match results, never exposing gallery data. We combine Differential Privacy (Opacus DP-SGD) with Homomorphic Encryption (TenSEAL CKKS) for the most comprehensive biometric privacy stack applied to this domain. A federated feedback loop enables the model to improve continuously from confirmed real-world matches. Evaluated on our custom surveillance dataset and CelebA under Non-IID federated distribution, our system achieves competitive identification accuracy while providing mathematically provable privacy guarantees.

---

*Document Version: 1.0 | August 2026*
