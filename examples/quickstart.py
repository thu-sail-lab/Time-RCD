#!/usr/bin/env python3
"""Minimal Time-RCD inference example on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from time_rcd import TimeRCDDetector


def main() -> None:
    rng = np.random.default_rng(42)
    length = 2048
    data = rng.normal(size=length)

    # Inject a simple anomaly spike.
    data[1000:1010] += 8.0

    detector = TimeRCDDetector.from_pretrained(variant="uni")
    scores = detector.predict(data)

    print(f"Input shape: {data.shape}")
    print(f"Score shape: {scores.shape}")
    print(f"Top-5 anomaly indices: {np.argsort(scores)[-5:][::-1]}")


if __name__ == "__main__":
    main()
