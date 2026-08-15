"""Caso de uso: extracao completa de landmarks para a camada silver (Secao 4 do plano).

Resiliente por design (achado em revisao de codigo): um processo de 300 videos
(~30-60min) que para inteiro no primeiro video problematico, sem log de qual
foi, e sem ter salvo nem o que ja funcionou, e um risco operacional real, nao
so teorico. Por isso:
- cada video roda em try/except proprio - uma falha e logada e pulada, nao
  derruba a extracao inteira;
- o manifest silver e reescrito periodicamente (nao so no final), entao um
  crash duro ainda deixa um manifest utilizavel do que foi processado ate ali;
- skip_existing pula videos cujo .npz ja existe em disco (recomputando so as
  estatisticas leves a partir do .npz salvo, sem rodar o MediaPipe de novo) -
  um restart depois de uma falha (ou so uma rerun deliberada) e barato.
"""
import csv
import time
from pathlib import Path

import numpy as np

from handwash.config.paths import (
    RAW_DATASET_DIR, HAND_LANDMARKER_TASK, SILVER_LANDMARKS_DIR, SILVER_MANIFEST,
    SILVER_DIR, BRONZE_MANIFEST,
)
from handwash.domain.gap_filling import interpolate_short_gaps, detection_rate
from handwash.infrastructure.landmark_extractor import extract_video_landmarks

SILVER_FIELDS = [
    "video_stem", "npz_relpath", "n_frames", "native_fps_used",
    "detection_rate_left", "detection_rate_right", "detection_rate_overall",
]
FAILURES_CSV = SILVER_DIR / "extraction_failures.csv"


def _read_bronze_manifest() -> list[dict]:
    with BRONZE_MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_silver_manifest(rows: list[dict]) -> None:
    SILVER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with SILVER_MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SILVER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_failures(failures: list[dict]) -> None:
    if not failures:
        return
    FAILURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_stem", "error"])
        writer.writeheader()
        writer.writerows(failures)


def _process_one_video(row: dict, skip_existing: bool) -> dict:
    video_stem = row["video_stem"]
    npz_relpath = f"{video_stem}.npz"
    npz_path = SILVER_LANDMARKS_DIR / npz_relpath

    if skip_existing and npz_path.exists():
        cached = np.load(npz_path)
        presence = cached["presence"]
        timestamps_sec = cached["timestamps_sec"]
    else:
        video_path = RAW_DATASET_DIR / row["raw_folder"] / row["filename"]
        raw = extract_video_landmarks(video_path, model_path=HAND_LANDMARKER_TASK)

        hand_lm, presence = interpolate_short_gaps(raw.hand_landmarks, raw.presence)
        world_lm, _ = interpolate_short_gaps(raw.world_landmarks, raw.presence)
        timestamps_sec = raw.timestamps_sec

        np.savez_compressed(
            npz_path,
            hand_landmarks=hand_lm,
            world_landmarks=world_lm,
            presence=presence,
            timestamps_sec=timestamps_sec,
        )

    last_ts = timestamps_sec[-1] if len(timestamps_sec) else 0.0
    return {
        "video_stem": video_stem,
        "npz_relpath": npz_relpath,
        "n_frames": presence.shape[0],
        "native_fps_used": round(presence.shape[0] / last_ts, 2) if last_ts > 0 else 0,
        "detection_rate_left": round(detection_rate(presence[:, 0:1]), 4),
        "detection_rate_right": round(detection_rate(presence[:, 1:2]), 4),
        "detection_rate_overall": round(detection_rate(presence), 4),
    }


def extract_all(limit: int | None = None, log_every: int = 10, skip_existing: bool = True) -> dict:
    bronze_rows = _read_bronze_manifest()
    if limit is not None:
        bronze_rows = bronze_rows[:limit]

    SILVER_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    silver_rows = []
    failures = []
    start_time = time.time()

    for i, row in enumerate(bronze_rows):
        try:
            silver_rows.append(_process_one_video(row, skip_existing))
        except Exception as exc:  # nao deixar um video ruim derrubar os outros 299
            failures.append({"video_stem": row["video_stem"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FALHA] {row['video_stem']}: {type(exc).__name__}: {exc}")

        if (i + 1) % log_every == 0 or (i + 1) == len(bronze_rows):
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * len(bronze_rows)
            print(
                f"[{i + 1}/{len(bronze_rows)}] {elapsed:.1f}s decorridos, "
                f"~{eta:.0f}s estimado total, {len(failures)} falha(s) ate agora"
            )
            _write_silver_manifest(silver_rows)  # escreve incrementalmente, nao so no final
            _write_failures(failures)

    _write_silver_manifest(silver_rows)
    _write_failures(failures)

    return {
        "n_videos_extracted": len(silver_rows),
        "n_failures": len(failures),
        "elapsed_sec": time.time() - start_time,
        "silver_manifest": str(SILVER_MANIFEST),
        "failures_csv": str(FAILURES_CSV) if failures else None,
    }


if __name__ == "__main__":
    import sys
    cli_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = extract_all(limit=cli_limit)
    print(summary)
