"""Preflight checker - run this before you submit.

It runs the exact contract we will run on our end: it calls your
predict.py the same way our grader will (as a subprocess, with
--data-dir and --out), against your own validation/ folder, and checks
that the output CSV has the right shape and vocabulary.

    python check_submission.py

If this prints "PREFLIGHT PASSED", our grader will be able to run your
code. This script checks plumbing only - it does NOT check whether your
predictions are any good. A model that is wrong about everything can still
pass preflight; that's by design, so passing this tells you nothing about
your score, only that we'll be able to compute one.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import config
from contract import (
    discover_class_names,
    find_entry_point,
    list_expected_filenames,
    TIMEOUT_SECONDS,
    validate_schema,
)

ROOT = Path(__file__).parent
VAL_DIR = os.path.join(config.DATA_ROOT, "validation")
LATEST_CHECKPOINT_PATH = ROOT / "runs" / "latest_checkpoint.txt"


def main():
    print(f"submission root: {ROOT}")
    print(f"checking against: {VAL_DIR}")

    try:
        entry_point = find_entry_point(ROOT)
    except (FileNotFoundError, ValueError) as e:
        print(f"PREFLIGHT FAILED: {e}")
        sys.exit(1)
    print(f"entry point: {entry_point}")

    class_names = discover_class_names(VAL_DIR)
    expected_files = list_expected_filenames(VAL_DIR)
    print(f"{len(expected_files)} images across {len(class_names)} classes: {class_names}")

    checkpoint_path = LATEST_CHECKPOINT_PATH.read_text().strip()

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / "preds.csv"
        try:
            start = time.time()
            proc = subprocess.run(
                [
                    sys.executable,
                    entry_point.name,
                    "--checkpoint",
                    checkpoint_path,
                    "--data-dir",
                    str(VAL_DIR),
                    "--out",
                    str(out_csv),
                ],
                cwd=str(entry_point.parent),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            elapsed = time.time() - start
        except Exception as e:
            print(f"PREFLIGHT FAILED: predict.py raised/timed out: {e}")
            sys.exit(1)

        print(f"predict.py exited {returncode} in {elapsed:.1f}s")
        if returncode != 0:
            print("--- stdout ---")
            print(stdout[-2000:])
            print("--- stderr ---")
            print(stderr[-2000:])
            print("PREFLIGHT FAILED: predict.py exited non-zero")
            sys.exit(1)

        result = validate_schema(out_csv, expected_files, class_names)

    if not result["ok"]:
        print(f"PREFLIGHT FAILED: {result['reason']}")
        sys.exit(1)

    print(f"wrote {result['n_rows']} valid predictions.")
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
