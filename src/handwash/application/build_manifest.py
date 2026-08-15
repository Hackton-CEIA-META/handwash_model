"""Caso de uso: construir o manifest bronze - fonte unica da verdade (Secao 4.1 do plano)."""
import csv
from pathlib import Path

from handwash.config.paths import RAW_DATASET_DIR, BRONZE_MANIFEST
from handwash.config.constants import SUBJECT_TO_GGROUP, SINK_BY_SUBJECT
from handwash.domain.filename_parsing import parse_video_filename
from handwash.domain.class_scheme import raw_folder_to_class, class_to_index
from handwash.domain.splits import split_for_subject, validate_split_integrity

MANIFEST_FIELDS = [
    "video_stem", "filename", "raw_folder", "subject_id", "g_group", "sink_name",
    "step_code", "merged_class", "merged_class_idx", "split",
]


def build_manifest(output_csv: Path = BRONZE_MANIFEST) -> list[dict]:
    validate_split_integrity()  # falha cedo se o dict de splits estiver quebrado

    rows = []
    for video_path in sorted(RAW_DATASET_DIR.glob("*/*.mp4")):
        raw_folder = video_path.parent.name
        parsed = parse_video_filename(video_path.name)
        merged_class = raw_folder_to_class(raw_folder)

        rows.append({
            "video_stem": video_path.stem,
            "filename": video_path.name,
            "raw_folder": raw_folder,
            "subject_id": parsed.subject_id,
            "g_group": SUBJECT_TO_GGROUP[parsed.subject_id],
            "sink_name": SINK_BY_SUBJECT[parsed.subject_id],
            "step_code": parsed.step_code,
            "merged_class": merged_class,
            "merged_class_idx": class_to_index(merged_class),
            "split": split_for_subject(parsed.subject_id),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    manifest_rows = build_manifest()
    print(f"{len(manifest_rows)} videos catalogados em {BRONZE_MANIFEST}")
