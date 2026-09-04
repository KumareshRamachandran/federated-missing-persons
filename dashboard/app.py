"""
dashboard/app.py

Streamlit-based Investigator Dashboard entrypoint.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

for p in [str(_THIS_DIR), str(_PROJECT_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)

sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_THIS_DIR))

try:
    from dashboard.dashboard.app import main
except ModuleNotFoundError:
    from dashboard.app import main

if __name__ == "__main__":
    main()


