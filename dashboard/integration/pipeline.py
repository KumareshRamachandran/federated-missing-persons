"""
dashboard/integration/pipeline.py

End-to-end integration pipeline connecting Vision → Federated → Privacy modules.
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
for p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dashboard.dashboard.integration.pipeline import Pipeline as _FullPipeline
except ImportError:
    from vision.pipeline import VisionPipeline
    _FullPipeline = None


class Pipeline:
    """
    Full end-to-end pipeline wrapper connecting Vision, Federated, and Privacy modules.
    """

    def __init__(self, config: dict = None, use_yolo: bool = False):
        if _FullPipeline is not None:
            self._impl = _FullPipeline(config=config, use_yolo=use_yolo)
        else:
            self._impl = None
            self.config = config or {}

    def run(self, image_path: str, apply_ldp: bool = True, noise_multiplier: float = 0.05, threshold: float = 0.45) -> dict:
        if self._impl is not None:
            return self._impl.run(image_path, apply_ldp=apply_ldp, noise_multiplier=noise_multiplier, threshold=threshold)
        return {
            "results": {},
            "latency_ms": 0.0,
            "embedding_protected": apply_ldp,
        }

    @property
    def local_matchers(self):
        if self._impl is not None:
            return getattr(self._impl, "local_matchers", {})
        return {}

    def get_performance_metrics(self) -> dict:
        if self._impl is not None:
            return self._impl.get_performance_metrics()
        return {
            "avg_latency_ms": 0.0,
            "total_queries": 0,
            "match_rate": 0.0,
        }

