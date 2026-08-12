"""
client/fl_client.py

Flower client — runs inside each organization node (Police, Hospital, Shelter, etc.)
Responsibilities:
  - Receive global model weights from coordinator
  - Perform local training on the org's private dataset
  - Apply Differential Privacy to gradients before sending back
  - Run local inference for search queries (Match/No-Match only)
"""

import flwr as fl
import torch
from face_engine.model import ArcFaceModel
from client.local_trainer import LocalTrainer
from client.local_matcher import LocalMatcher


class OrgFLClient(fl.client.NumPyClient):
    """Flower NumPy client for a single organization node."""

    def __init__(self, node_id: str, data_dir: str):
        self.node_id = node_id
        self.model = ArcFaceModel()
        self.trainer = LocalTrainer(self.model, data_dir)
        self.matcher = LocalMatcher(self.model, data_dir)

    def get_parameters(self, config):
        """Return current local model weights as a list of NumPy arrays."""
        # TODO: Extract model parameters via model.state_dict()
        pass

    def fit(self, parameters, config):
        """Receive global weights, train locally, return updated weights."""
        # TODO: Load received parameters into model
        # TODO: Run local training via self.trainer.train()
        # TODO: Apply DP noise to gradients
        # TODO: Return updated weights, num_samples, metrics
        pass

    def evaluate(self, parameters, config):
        """Evaluate model on local validation set."""
        # TODO: Load parameters, run evaluation, return loss and metrics
        pass


def start_client(node_id: str, data_dir: str, server_address: str = "localhost:8080"):
    """Launch this org node as a Flower client."""
    client = OrgFLClient(node_id, data_dir)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    import sys
    node_id = sys.argv[1] if len(sys.argv) > 1 else "node_1"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else f"data/nodes/{node_id}"
    start_client(node_id, data_dir)
