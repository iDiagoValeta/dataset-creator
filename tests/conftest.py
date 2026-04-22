"""Pytest configuration: make `pipeline/` importable without installing the package."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
