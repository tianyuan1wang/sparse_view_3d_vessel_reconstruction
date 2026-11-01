#!/usr/bin/env python3
"""Run the proposed reconstruction entry point from any working directory."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
sys.path.insert(0, str(SOURCE))
runpy.run_path(str(SOURCE / "run_compare_full128_prior_direct128_test50.py"), run_name="__main__")

