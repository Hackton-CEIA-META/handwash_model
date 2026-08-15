"""Caso de uso: medir fps/resolucao reais de todos os videos brutos (Secao 1 / verificacao do plano)."""
from collections import Counter
from pathlib import Path
import csv

from handwash.config.paths import RAW_DATASET_DIR, BRONZE_PROBE_REPORT
from handwash.infrastructure.video_probe import probe_video


def iter_raw_videos() -> list[Path]:
    return sorted(RAW_DATASET_DIR.glob("*/*.mp4"))


def run_probe(output_csv: Path = BRONZE_PROBE_REPORT) -> dict:
    videos = iter_raw_videos()
    if not videos:
        raise FileNotFoundError(f"Nenhum .mp4 encontrado em {RAW_DATASET_DIR}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fps_counter: Counter = Counter()
    resolution_counter: Counter = Counter()
    rows = []

    for video_path in videos:
        result = probe_video(video_path)
        fps_rounded = round(result.fps, 2)
        fps_counter[fps_rounded] += 1
        resolution_counter[(result.width, result.height)] += 1
        rows.append({
            "raw_folder": video_path.parent.name,
            "filename": video_path.name,
            "fps": result.fps,
            "frame_count": result.frame_count,
            "width": result.width,
            "height": result.height,
            "duration_sec": result.duration_sec,
        })

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "total_videos": len(videos),
        "fps_distribution": dict(fps_counter),
        "resolution_distribution": {f"{w}x{h}": c for (w, h), c in resolution_counter.items()},
        "output_csv": str(output_csv),
    }


if __name__ == "__main__":
    summary = run_probe()
    print(summary)
