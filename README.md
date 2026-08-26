# Privacy-Preserving Federated Learning for Missing Person Identification
**Domain:** Computer Vision · Federated Learning · Biometric Privacy  
**Academic Year:** Fall Semester 2026–2027  
**Guide:** Kanhaiya Sharma (70880)

---

## Team & Module Ownership

| Folder | Member | Reg. No | Responsibility |
|--------|--------|---------|----------------|
| `vision/` | G N Lokesh | 23BCE9603 | YOLO detection, ArcFace embeddings, dataset preprocessing |
| `federated/` | R Kumaresh | 23BCE9585 | Flower FL framework, FedAvg, edge node communication |
| `privacy/` | K Kishore | 23BCE9746 | DP (Opacus), SMPC/HE (TenSEAL), privacy evaluation |
| `dashboard/` | Aswin Maheswaran | 23BCE8540 | Streamlit UI, module integration, performance testing |

---

## Project Structure

```
federated-missing-persons/
│
├── vision/                        ← G N LOKESH (Computer Vision)
│   ├── yolo_detector.py           # YOLOv8 human detection from surveillance frames
│   ├── face_detector.py           # MTCNN face detection & alignment (112×112)
│   ├── arcface_model.py           # ArcFace iResNet50 model wrapper
│   ├── embedder.py                # Full pipeline: image → 512-d embedding
│   ├── augmentation.py            # Simulate CCTV conditions (low-light, blur, occlusion)
│   └── dataset/
│       ├── download_celeba.py     # Download CelebA (primary training dataset)
│       ├── download_lfw.py        # Download LFW (evaluation dataset)
│       └── partition_data.py      # Non-IID split across federated nodes
│
├── federated/                     ← R KUMARESH (Federated Learning)
│   ├── coordinator/
│   │   ├── server.py              # Flower FL server (starts federation)
│   │   ├── strategy.py            # Custom FedAvg + DP aggregation strategy
│   │   ├── model_manager.py       # Global model versioning & accuracy tracking
│   │   ├── query_router.py        # Broadcast query embedding → collect results
│   │   └── api.py                 # FastAPI REST endpoints (search, status, accuracy)
│   └── client/
│       ├── fl_client.py           # Flower NumPy client for each org node
│       ├── local_trainer.py       # Local training loop (Opacus DP-SGD)
│       └── local_matcher.py       # Privacy-preserving inference (returns only Match/No-Match)
│
├── privacy/                       ← K KISHORE (Cryptography & Privacy)
│   ├── dp_utils.py                # Gradient clipping, Gaussian noise, epsilon budget
│   ├── smpc.py                    # TenSEAL CKKS Homomorphic Encryption for model weights
│   ├── secure_aggregation.py      # Simulated Secure Aggregation (mask-cancel protocol)
│   └── privacy_evaluator.py       # Measure epsilon, accuracy drop, MIA resistance
│
├── dashboard/                     ← ASWIN MAHESWARAN (UI & Integration)
│   ├── app.py                     # Streamlit investigator dashboard (main GUI)
│   ├── integration/
│   │   └── pipeline.py            # End-to-end module integration (Vision→FL→Privacy→UI)
│   └── evaluation/
│       ├── metrics.py             # Rank-1 accuracy, FMR, FNMR, ROC, EER
│       └── benchmark.py           # Centralized vs Federated comparison
│
├── data/                          ← Shared (not committed to git)
│   ├── raw/                       # Downloaded CelebA, LFW datasets
│   ├── nodes/                     # Partitioned per-org galleries
│   │   ├── node_police/
│   │   ├── node_hospital/
│   │   └── node_ngo/
│   └── query_set/                 # Held-out query images (simulates "missing person" photo)
│
├── shared/
│   ├── configs/config.yaml        # All system config (server addr, DP params, model path)
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_federated_training.ipynb
│   │   └── 03_evaluation_results.ipynb
│   └── tests/
│       ├── test_face_engine.py
│       └── test_local_matcher.py
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Setup environment
conda create -n fedmissing python=3.10
conda activate fedmissing
pip install -r requirements.txt

# 2. Download & partition datasets (LOKESH)
python vision/dataset/download_celeba.py --output_dir data/raw/celeba
python vision/dataset/download_lfw.py --output_dir data/raw/lfw
python vision/dataset/partition_data.py --celeba_dir data/raw/celeba --output_dir data/nodes

# 3. Start FL coordinator (KUMARESH)
python -m federated.coordinator.server

# 4. Start org node clients in separate terminals (KUMARESH)
python -m federated.client.fl_client node_police   data/nodes/node_police
python -m federated.client.fl_client node_hospital data/nodes/node_hospital
python -m federated.client.fl_client node_ngo      data/nodes/node_ngo

# 5. Run the dashboard (ASWIN)
streamlit run dashboard/app.py
```

---

## Datasets

| Dataset | Size | Use |
|---------|------|-----|
| CelebA | ~1.63 GB, 202K images, 10,177 identities | Primary FL training data |
| LFW | ~180 MB, 13K images, 5,749 identities | Evaluation & verification |
| Custom CCTV Dataset | 780 images (240 positive, 540 decoy) | YOLO + embedding stress test |

---

## Key Papers
- McMahan et al. (2017) — FedAvg: Communication-Efficient Federated Learning
- Deng et al. (2019) — ArcFace: Additive Angular Margin Loss
- Abadi et al. (2016) — Deep Learning with Differential Privacy (Opacus)
- Bonawitz et al. (2017) — Practical Secure Aggregation
- Cheon et al. (2017) — CKKS Homomorphic Encryption (TenSEAL)
