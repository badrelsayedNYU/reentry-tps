"""Shared command-line handling for the study scripts."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--materials", default=ROOT / "data" / "materials.yaml", type=Path)
    ap.add_argument("--baseline", default=ROOT / "data" / "baseline.yaml", type=Path)
    ap.add_argument("--out", default=ROOT, type=Path, help="writes into <out>/figures and <out>/results")
    args = ap.parse_args()
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    return args


def load(args, shield=None):
    from tps.materials import MissingDataError
    from tps.sizing import load_case

    try:
        return load_case(args.materials, args.baseline, shield=shield)
    except MissingDataError as err:
        sys.exit(f"cannot run yet: {err}")
