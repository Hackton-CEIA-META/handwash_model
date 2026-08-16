"""Caso de uso: construir a camada gold (Secao 4.4-5 do plano) - janelas normalizadas,
prontas pra treino, com o split por sujeito ja aplicado.

Pipeline por video: le o .npz silver (world_landmarks, presence, timestamps_sec) ->
reamostra da grade nativa pra 15fps por timestamp (linear p/ landmarks, nearest p/
presence - ver domain.resampling) -> normaliza por mao (domain.normalization) -> fecha a
brecha presence/normalizacao na fronteira de gaps longos -> monta o vetor de 128
features/frame -> fatia em janelas de 30 passos, com stride por split (0.5s treino, 1.0s
val/teste - Secao 3/4 do plano) -> anexa metadados (sujeito, classe, g-group, tempo de
inicio) pra rastreabilidade e voto majoritario por video (Secao 6).

`manifest.csv` (bronze) e a fonte da verdade pra split/classe por video - nao
re-derivamos nada disso aqui. `data/gold/manifest.csv` vira a mesma fonte da verdade
pras janelas: guarda 'split_row_idx', a posicao exata da janela dentro do array salvo em
data/gold/{split}/windows.npz, pra juntar metadados e arrays sem duplicar dado nos dois
lugares.
"""
import csv
import json

import numpy as np

from handwash.config.constants import (
    CLASSES, TARGET_FPS, TRAIN_STRIDE_SEC, VAL_TEST_STRIDE_SEC, WINDOW_FEATURE_DIM, WINDOW_STEPS,
)
from handwash.config.paths import BRONZE_MANIFEST, GOLD_CLASSES_JSON, GOLD_DIR, SILVER_LANDMARKS_DIR
from handwash.domain.class_scheme import class_to_index
from handwash.domain.normalization import normalize_and_mask
from handwash.domain.resampling import resample_linear, resample_nearest, target_grid
from handwash.domain.splits import assert_no_cross_split_subjects, validate_split_integrity
from handwash.domain.windowing import slice_windows, window_start_indices

GOLD_MANIFEST_FIELDS = [
    "split_row_idx", "video_stem", "subject_id", "g_group", "split",
    "class_name", "class_idx", "window_idx", "start_frame", "start_time_sec",
]
SPLITS = ("train", "val", "test")
# 0.5s nao divide exatamente em passos de 15fps (0.5*15=7.5) - arredonda pro inteiro mais
# proximo (8 passos, ~0.53s real). 1.0s divide exato (15 passos). Diferenca esperada, nao bug.
STRIDE_SEC_BY_SPLIT = {"train": TRAIN_STRIDE_SEC, "val": VAL_TEST_STRIDE_SEC, "test": VAL_TEST_STRIDE_SEC}


def _read_bronze_manifest() -> list[dict]:
    with BRONZE_MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_video_features(video_stem: str, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (features, target_timestamps). features: float32[n_resampled, 128].

    normalize=False retorna landmarks BRUTOS (so reamostrados, sem normalize_and_mask) -
    usado pelo export TFLite (Secao 7 do plano, application.export_tflite): o grafo
    exportado embute a normalizacao internamente (CLAUDE.md: "nao reimplementada em
    Kotlin"), entao o dataset representativo de calibracao INT8 precisa ver a MESMA
    distribuicao bruta que o grafo vai receber em producao, nao o dado ja normalizado
    que build_windows() usa pra treino."""
    npz_path = SILVER_LANDMARKS_DIR / f"{video_stem}.npz"
    with np.load(npz_path) as data:
        world_landmarks = data["world_landmarks"]
        presence = data["presence"]
        timestamps_sec = data["timestamps_sec"]

    target_ts = target_grid(float(timestamps_sec[-1]), TARGET_FPS)
    world_resampled = resample_linear(world_landmarks, timestamps_sec, target_ts)
    presence_resampled = resample_nearest(presence, timestamps_sec, target_ts)

    landmarks = normalize_and_mask(world_resampled, presence_resampled) if normalize else world_resampled

    n_resampled = landmarks.shape[0]
    flat_landmarks = landmarks.reshape(n_resampled, -1)  # (n_resampled, 126)
    features = np.concatenate([flat_landmarks, presence_resampled.astype(np.float32)], axis=-1)

    if features.shape[-1] != WINDOW_FEATURE_DIM:
        raise ValueError(
            f"Feature dim inesperada para {video_stem}: {features.shape[-1]} != {WINDOW_FEATURE_DIM}"
        )
    return features.astype(np.float32), target_ts


def _windows_for_video(bronze_row: dict) -> tuple[np.ndarray, list[dict]]:
    video_stem = bronze_row["video_stem"]
    split = bronze_row["split"]
    class_name = bronze_row["merged_class"]
    class_idx = class_to_index(class_name)  # constante pro video inteiro - calcular 1x, nao por janela

    features, target_ts = build_video_features(video_stem)
    stride_steps = round(STRIDE_SEC_BY_SPLIT[split] * TARGET_FPS)
    starts = window_start_indices(features.shape[0], WINDOW_STEPS, stride_steps)
    windows = slice_windows(features, starts, WINDOW_STEPS)

    rows = [
        {
            "video_stem": video_stem,
            "subject_id": int(bronze_row["subject_id"]),
            "g_group": bronze_row["g_group"],
            "split": split,
            "class_name": class_name,
            "class_idx": class_idx,
            "window_idx": window_idx,
            "start_frame": start,
            "start_time_sec": round(float(target_ts[start]), 4),
        }
        for window_idx, start in enumerate(starts)
    ]
    return windows, rows


def build_windows(limit: int | None = None) -> dict:
    validate_split_integrity()  # falha cedo se o dict de splits estiver quebrado

    bronze_rows = _read_bronze_manifest()
    if limit is not None:
        bronze_rows = bronze_rows[:limit]

    per_split_X: dict[str, list[np.ndarray]] = {split: [] for split in SPLITS}
    manifest_rows: list[dict] = []
    videos_with_zero_windows: list[str] = []

    for bronze_row in bronze_rows:
        windows, rows = _windows_for_video(bronze_row)
        if windows.shape[0] == 0:
            videos_with_zero_windows.append(bronze_row["video_stem"])
            continue

        per_split_X[bronze_row["split"]].append(windows)
        manifest_rows.extend(rows)

    assert_no_cross_split_subjects(manifest_rows)  # Secao 3 do plano: assert contra o dado real

    # split_row_idx e derivado numa unica passada por split (filtrar manifest_rows + enumerate)
    # em vez de um contador incrementado dentro do loop acima. A ordem de manifest_rows por
    # split ja bate com a ordem de concatenacao de per_split_X[split] por construcao (os dois
    # vem do mesmo loop, na mesma iteracao de video) - derivar aqui deixa isso obviamente
    # correto em vez de depender de dois acumuladores ficarem sincronizados manualmente
    # (revisao de codigo: essa era exatamente a fragilidade que quebraria se o loop acima
    # fosse paralelizado no futuro).
    rows_by_split = {split: [r for r in manifest_rows if r["split"] == split] for split in SPLITS}
    for split_rows in rows_by_split.values():
        for split_row_idx, row in enumerate(split_rows):
            row["split_row_idx"] = split_row_idx

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    counts_by_split = {}
    counts_by_split_class = {}
    for split in SPLITS:
        X = (
            np.concatenate(per_split_X[split], axis=0)
            if per_split_X[split]
            else np.zeros((0, WINDOW_STEPS, WINDOW_FEATURE_DIM), dtype=np.float32)
        )
        y = np.array([row["class_idx"] for row in rows_by_split[split]], dtype=np.int64)

        split_dir = GOLD_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_dir / "windows.npz", X=X, y=y)

        counts_by_split[split] = int(X.shape[0])
        class_counts: dict[str, int] = {}
        for row in rows_by_split[split]:
            class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1
        counts_by_split_class[split] = class_counts

    manifest_path = GOLD_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GOLD_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    GOLD_CLASSES_JSON.write_text(json.dumps(CLASSES, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "n_videos_processed": len(bronze_rows),
        "n_total_windows": sum(counts_by_split.values()),
        "counts_by_split": counts_by_split,
        "counts_by_split_class": counts_by_split_class,
        "videos_with_zero_windows": videos_with_zero_windows,
        "gold_manifest": str(manifest_path),
        "gold_classes_json": str(GOLD_CLASSES_JSON),
    }


if __name__ == "__main__":
    import sys
    cli_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    summary = build_windows(limit=cli_limit)
    print(summary)
