"""Grab one frame with DXGI (and optionally WGC) and save it as a PNG."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from ocrtrans.capture import CaptureManager, DxgiBackend, WgcBackend
from ocrtrans.geometry import Rect

FULL = Rect(0, 0, 10_000, 10_000)  # numpy slicing clamps to the frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", help="window title for the WGC fallback")
    parser.add_argument("--out", default="capture_smoke.png")
    args = parser.parse_args()

    dxgi = DxgiBackend()
    frame = None
    for _ in range(40):
        candidate = dxgi.grab(FULL)
        if candidate is not None and float(candidate.mean()) >= 2.0:
            frame = candidate
            break
        time.sleep(0.05)
    if frame is None:
        print("DXGI returned no frame")
        return 1
    print(f"DXGI frame {frame.shape} mean={frame.mean():.1f} dtype={frame.dtype}")
    cv2.imwrite(args.out, frame)
    print(f"saved {Path(args.out).resolve()}")

    if args.window:
        manager = CaptureManager(dxgi, WgcBackend(window_name=args.window))
        for _ in range(20):
            frame, health = manager.grab(FULL)
            if frame is not None:
                break
            time.sleep(0.05)
        print(f"WGC fallback health={health.name} shape={None if frame is None else frame.shape}")
        manager.close()
    else:
        dxgi.close()

    return 1 if frame is None or float(frame.mean()) < 2.0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
