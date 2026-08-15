"""Caso de uso: checagem de viabilidade do MediaPipe numa amostra (Secao 1.3 do plano - GATE).

Amostra videos de cada uma das 12 pastas brutas (nao das 7 classes finais) para
cobrir toda a variacao visual, incluindo sujeitos do G05 (setup de camera
visivelmente diferente: 1920x1080 vs 720x480 nos demais grupos) - a rotacao de
sujeitos entre pastas vem de domain.sampling (ver docstring la sobre o bug que
motivou extrair essa logica pra uma funcao pura testavel).

O gate usa at_least_one_hand_rate, nao a media simples sobre os 2 slots de mao:
varios passos da tecnica OMS entrelacam/sobrepoem as maos por definicao, o que
derruba a media combinada mesmo com deteccao saudavel. both_hands_rate e
reportado, mas so como informacao de diagnostico.
"""
from collections import defaultdict
from pathlib import Path
import csv

from handwash.config.paths import RAW_DATASET_DIR, HAND_LANDMARKER_TASK, BRONZE_DIR
from handwash.config.constants import RAW_FOLDER_TO_CLASS, VIABILITY_MIN_AT_LEAST_ONE_HAND_RATE
from handwash.domain.filename_parsing import parse_video_filename
from handwash.domain.gap_filling import at_least_one_hand_rate, both_hands_rate, detection_rate
from handwash.domain.sampling import sample_subjects_for_folder
from handwash.infrastructure.landmark_extractor import extract_video_landmarks

VIABILITY_REPORT_CSV = BRONZE_DIR / "viability_check_report.csv"

_NON_G05_SUBJECT_IDS = list(range(1, 18))
_G05_SUBJECT_IDS = list(range(18, 26))


def _sample_videos_per_raw_folder(n_per_folder: int = 4) -> list[Path]:
    samples = []
    for folder_idx, raw_folder in enumerate(sorted(RAW_FOLDER_TO_CLASS.keys())):
        videos = sorted((RAW_DATASET_DIR / raw_folder).glob("*.mp4"))
        if not videos:
            continue

        chosen_ids = sample_subjects_for_folder(
            folder_idx, _NON_G05_SUBJECT_IDS, _G05_SUBJECT_IDS, n_per_folder
        )
        by_id = {parse_video_filename(v.name).subject_id: v for v in videos}
        samples.extend(by_id[sid] for sid in chosen_ids if sid in by_id)
    return samples


def run_viability_check(n_per_folder: int = 4) -> dict:
    videos = _sample_videos_per_raw_folder(n_per_folder)
    rows = []

    for video_path in videos:
        raw_folder = video_path.parent.name
        result = extract_video_landmarks(video_path, model_path=HAND_LANDMARKER_TASK)
        rows.append({
            "raw_folder": raw_folder,
            "merged_class": RAW_FOLDER_TO_CLASS[raw_folder],
            "filename": video_path.name,
            "n_frames": result.presence.shape[0],
            "at_least_one_hand_rate": at_least_one_hand_rate(result.presence),
            "both_hands_rate": both_hands_rate(result.presence),
            "detection_rate_combined": detection_rate(result.presence),
        })

    VIABILITY_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with VIABILITY_REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_folder_one_hand = defaultdict(list)
    by_folder_both_hands = defaultdict(list)
    for row in rows:
        by_folder_one_hand[row["raw_folder"]].append(row["at_least_one_hand_rate"])
        by_folder_both_hands[row["raw_folder"]].append(row["both_hands_rate"])

    one_hand_summary = {f: sum(r) / len(r) for f, r in by_folder_one_hand.items()}
    both_hands_summary = {f: sum(r) / len(r) for f, r in by_folder_both_hands.items()}
    failing_folders = {
        f: rate for f, rate in one_hand_summary.items() if rate < VIABILITY_MIN_AT_LEAST_ONE_HAND_RATE
    }

    return {
        "n_videos_checked": len(rows),
        "gate_metric": "at_least_one_hand_rate",
        "gate_threshold": VIABILITY_MIN_AT_LEAST_ONE_HAND_RATE,
        "overall_mean_at_least_one_hand_rate": sum(r["at_least_one_hand_rate"] for r in rows) / len(rows),
        "overall_mean_both_hands_rate": sum(r["both_hands_rate"] for r in rows) / len(rows),
        "per_raw_folder_at_least_one_hand_rate": one_hand_summary,
        "per_raw_folder_both_hands_rate_informational": both_hands_summary,
        "folders_below_threshold": failing_folders,
        "gate_passed": len(failing_folders) == 0,
        "report_csv": str(VIABILITY_REPORT_CSV),
    }


if __name__ == "__main__":
    summary = run_viability_check()
    print(summary)
