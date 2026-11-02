"""Accuracy on pairwise probabilities. A prediction is correct when prob > 0.5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap


def binary_metrics(probs) -> dict:
    values = np.asarray(probs, dtype=np.float64)
    correct = values > 0.5
    accuracy = float(correct.mean()) if len(values) else 0.0
    if len(values) < 2:
        low = high = accuracy
    else:
        interval = bootstrap((correct.astype(float),), np.mean, confidence_level=0.95, method="basic")
        low = float(interval.confidence_interval.low)
        high = float(interval.confidence_interval.high)
    wrong = ~correct
    return {
        "n": int(len(values)),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "ci95_low": low,
        "ci95_high": high,
        "n_wrong": int(wrong.sum()),
        "n_near_half": int(np.sum(np.abs(values - 0.5) <= 0.05)),
        "n_confident_wrong": int(np.sum(wrong & (np.abs(values - 0.5) > 0.2))),
        "mean_prob": float(values.mean()) if len(values) else 0.0,
    }


def summarize_prediction_file(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = binary_metrics(payload["probs"])
    metrics["prediction_file"] = str(Path(path).resolve())
    metrics["checkpoint"] = payload.get("ckpt_dir", "")
    stored = payload.get("metrics") or {}
    metrics["stored_accuracy"] = stored.get("accuracy")
    return metrics


def write_summary(metrics: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"n={metrics['n']} accuracy={metrics['accuracy_percent']:.2f}% "
        f"ci95=[{metrics['ci95_low'] * 100:.2f}, {metrics['ci95_high'] * 100:.2f}] "
        f"mean_prob={metrics['mean_prob']:.3f}"
    )
    print(f"summary {destination}")
