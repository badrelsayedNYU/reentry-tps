"""Regenerate every figure and results table: validation, baseline, convergence, sweeps.

Usage: python scripts/make_figures.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ("validate.py", "run_baseline.py", "convergence.py", "sweeps.py"):
    print(f"== {script}")
    subprocess.run([sys.executable, str(HERE / script), *sys.argv[1:]], check=True)
