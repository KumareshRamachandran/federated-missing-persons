"""
test.py — Root test runner for Federated Missing Persons Project
=================================================================

Runs all computer vision, dataset, and pipeline unit/integration tests.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision.test import run_all_tests

if __name__ == "__main__":
    run_all_tests()
