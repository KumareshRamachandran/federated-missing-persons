# Privacy-Preserving Federated AI System for Missing Person Identification

## Project Structure

```
federated-missing-persons/
├── coordinator/          # Central FL server & query router
├── client/               # Organization node FL client & local matcher
├── face_engine/          # Face detection, embedding, model
├── privacy/              # Differential Privacy & Secure Aggregation
├── data_scripts/         # Dataset download & partitioning
├── evaluation/           # Metrics & centralized vs federated benchmark
├── notebooks/            # EDA, training experiments, results
├── tests/                # Unit tests
├── configs/              # config.yaml
├── frontend/             # React demo UI (investigator search interface)
├── docs/                 # Documentation
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and partition dataset
python data_scripts/download_lfw.py --output_dir data/raw/lfw
python data_scripts/partition_data.py --lfw_dir data/raw/lfw --output_dir data/nodes

# 4. Start coordinator server
python -m coordinator.server

# 5. Start org node clients (in separate terminals)
python -m client.fl_client node_police data/nodes/node_police
python -m client.fl_client node_hospital data/nodes/node_hospital
python -m client.fl_client node_ngo data/nodes/node_ngo
```

## Dataset
- **LFW** (Labeled Faces in the Wild) — Federated simulation
- **CASIA-WebFace** — Pre-training the ArcFace backbone

## Key Papers
- McMahan et al. (2017) — FedAvg
- Deng et al. (2019) — ArcFace
- Abadi et al. (2016) — Deep Learning with Differential Privacy
