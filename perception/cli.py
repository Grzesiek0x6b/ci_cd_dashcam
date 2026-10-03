"""CLI: uruchamia potok (DETR -> śledzenie -> głębia) na pliku wideo.

Użycie: perception VIDEO [--out wynik.mp4] [--json wynik.json] [--no-depth] [--device cpu]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from perception.pipeline import Record, run
from perception.video import load_frames


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, 3))  # type: ignore[return-value]


def draw(frame: np.ndarray, records: list[Record], id2label: dict[int, str]) -> np.ndarray:
    """Rysuje skrzynki z ID, klasą i `near` na kopii klatki RGB."""
    vis = frame.copy()
    for r in records:
        x1, y1, x2, y2 = (int(v) for v in r.box)
        col = _color(r.id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        text = f"#{r.id} {id2label.get(r.label, r.label)}" + (f" {r.near:.2f}" if r.near is not None else "")
        cv2.putText(vis, text, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return vis


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    if path.suffix.lower() not in ("", ".mp4"):
        raise ValueError("Plik wyjściowy musi mieć rozszerzenie .mp4")
    if not path.suffix:
        path = path.with_suffix(".mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Nie można otworzyć pliku wyjściowego: {path}")
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="perception", description=__doc__)
    ap.add_argument("video", type=Path, help="plik wideo (dashcam)")
    ap.add_argument("--out", type=Path, help="zapisz wideo z naniesionymi skrzynkami (mp4)")
    ap.add_argument("--json", type=Path, help="zapisz rekordy (klatka, id, klasa, box, near) do JSON")
    ap.add_argument("--max-frames", type=int, default=180)
    ap.add_argument("--stride", type=int, default=4, help="bierz co N-tą klatkę")
    ap.add_argument("--width", type=int, default=960, help="szerokość po przeskalowaniu")
    ap.add_argument("--no-depth", action="store_true", help="pomiń estymację głębi (szybciej)")
    ap.add_argument("--device", default=None, help="cpu/mps/cuda; domyślnie auto")
    args = ap.parse_args(argv)

    try:
        frames, fps = load_frames(args.video, args.max_frames, args.stride, args.width)
    except FileNotFoundError:
        print(f"Brak pliku: {args.video}", file=sys.stderr)
        return 2
    if not frames:
        print(f"Nie udało się wczytać klatek z {args.video}", file=sys.stderr)
        return 2

    from perception.depth import DepthEstimator
    from perception.detect import Detector

    print(f"Klatek: {len(frames)}, ładuję modele...", file=sys.stderr)
    detector = Detector(device=args.device)
    depth = None if args.no_depth else DepthEstimator(device=args.device)
    res = run(frames, detector, depth)

    by_frame: dict[int, list[Record]] = {}
    for r in res.records:
        by_frame.setdefault(r.frame, []).append(r)
    for fi in range(len(frames)):
        recs = by_frame.get(fi, [])
        desc = ", ".join(f"#{r.id} {detector.id2label.get(r.label, r.label)}" for r in recs) or "-"
        print(f"klatka {fi:4d}: {desc}")
    ms = 1000 * float(np.median(res.seconds_per_frame))
    print(
        f"\nUnikalne tory: {len({r.id for r in res.records})}, mediana {ms:.0f} ms/klatkę ({detector.device})"
    )

    if args.json:
        rows = [
            {
                "frame": r.frame,
                "id": r.id,
                "label": detector.id2label.get(r.label, str(r.label)),
                "box": [round(float(v), 1) for v in r.box],
                "near": None if r.near is None else round(float(r.near), 4),
            }
            for r in res.records
        ]
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"JSON: {args.json}")
    if args.out:
        try:
            out_path = write_video(
                args.out, [draw(f, by_frame.get(i, []), detector.id2label) for i, f in enumerate(frames)], fps
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Nie udało się zapisać wideo: {exc}", file=sys.stderr)
            return 2
        print(f"Wideo: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
