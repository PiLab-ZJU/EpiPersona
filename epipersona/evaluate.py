"""Write a metric summary for one inference prediction file."""

from __future__ import annotations

import argparse

from epipersona.metrics import summarize_prediction_file, write_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize reward-model predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = summarize_prediction_file(args.predictions)
    output = args.output
    if output is None:
        from pathlib import Path
        path = Path(args.predictions)
        output = path.with_name(path.stem + ".summary.json")
    write_summary(metrics, output)


if __name__ == "__main__":
    main()
