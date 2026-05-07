#!/usr/bin/env python3
"""
ARIAN Wildfire Prediction — Full Pipeline Runner
═════════════════════════════════════════════════
Executes all 6 notebooks sequentially with progress tracking.

Usage:
    python pipeline.py              # run all notebooks
    python pipeline.py --from 3     # resume from NB03 onward
    python pipeline.py --only 4     # run only NB04
"""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm

NOTEBOOKS_DIR = Path(__file__).parent / "notebooks"

NOTEBOOKS = [
    ("01_Data_Ingestion.ipynb",          "NB01 — Data Ingestion"),
    ("02_EDA_FeatureEngineering.ipynb",  "NB02 — EDA & Feature Engineering"),
    ("03_Weather_TimeSeries.ipynb",      "NB03 — Weather Time-Series"),
    ("04_Wildfire_Detection.ipynb",      "NB04 — Wildfire Detection"),
    ("05_Risk_Prediction_Dashboard.ipynb","NB05 — Risk Prediction Dashboard"),
    ("06_Climate_Report.ipynb",          "NB06 — Climate Report"),
]

KERNEL_NAME = "python3"
TIMEOUT = None  # no timeout — some notebooks may run for hours


def ensure_gee_auth():
    """Prompt for GEE authentication before headless notebook execution."""
    try:
        import ee
        ee.Initialize(project="manheim-fire-detection")
        print("  ✔  GEE credentials found — already authenticated.\n")
    except Exception:
        print("  ⚠  GEE not authenticated. Running ee.Authenticate()...")
        print("     Paste the token when prompted.\n")
        import ee
        ee.Authenticate()
        ee.Initialize(project="manheim-fire-detection")
        print("  ✔  GEE authenticated successfully.\n")


def run_notebook(path: Path, label: str) -> dict:
    """Execute a single notebook in-place. Returns stats dict."""
    nb = nbformat.read(path, as_version=4)

    # Skip cells that require interactive input (e.g. GEE token paste)
    nb.cells = [
        c for c in nb.cells
        if not (c.cell_type == "code" and "ee.Authenticate()" in c.source)
    ]

    ep = ExecutePreprocessor(
        timeout=TIMEOUT,
        kernel_name=KERNEL_NAME,
        cwd=str(path.parent),
    )

    n_cells = len(nb.cells)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")

    t0 = time.time()
    ep.preprocess(nb)
    elapsed = time.time() - t0

    # save executed notebook (outputs embedded)
    nbformat.write(nb, path)

    return {
        "label": label,
        "cells": n_cells,
        "code_cells": code_cells,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ARIAN pipeline notebooks sequentially.")
    parser.add_argument("--from", dest="start", type=int, default=1,
                        help="Start from notebook N (1-6)")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only notebook N (1-6)")
    args = parser.parse_args()

    if args.only:
        selected = [(f, l) for f, l in NOTEBOOKS if f.startswith(f"{args.only:02d}_")]
        if not selected:
            print(f"ERROR: No notebook found for --only {args.only}")
            sys.exit(1)
    else:
        selected = [(f, l) for f, l in NOTEBOOKS if int(f[:2]) >= args.start]

    if not selected:
        print("No notebooks to run.")
        sys.exit(0)

    print("╔══════════════════════════════════════════════════╗")
    print("║   ARIAN Wildfire Prediction — Pipeline Runner   ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Notebooks : {len(selected)} of {len(NOTEBOOKS)}")
    print(f"  Directory : {NOTEBOOKS_DIR}")
    print()

    ensure_gee_auth()

    results = []
    overall_t0 = time.time()

    pbar = tqdm(selected, unit="nb", desc="Pipeline", ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for filename, label in pbar:
        path = NOTEBOOKS_DIR / filename
        if not path.exists():
            pbar.write(f"  ⚠  SKIP  {label} — file not found")
            continue

        pbar.set_postfix_str(label, refresh=True)
        pbar.write(f"  ▶  START  {label}")

        try:
            stats = run_notebook(path, label)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(stats["elapsed"]))
            pbar.write(f"  ✔  DONE   {label}  ({stats['code_cells']} code cells, {elapsed_str})")
            results.append(stats)
        except Exception as exc:
            pbar.write(f"  ✖  FAIL   {label}")
            pbar.write(f"           {type(exc).__name__}: {exc}")
            print("\nPipeline stopped due to error above.")
            sys.exit(1)

    overall_elapsed = time.time() - overall_t0
    overall_str = time.strftime("%H:%M:%S", time.gmtime(overall_elapsed))

    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│              Pipeline Summary                    │")
    print("├──────────────────────────────────────────────────┤")
    for r in results:
        t = time.strftime("%H:%M:%S", time.gmtime(r["elapsed"]))
        print(f"│  ✔ {r['label']:<36s} {t:>8s} │")
    print("├──────────────────────────────────────────────────┤")
    print(f"│  Total: {len(results)} notebooks{' '*22}{overall_str:>8s} │")
    print("└──────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
