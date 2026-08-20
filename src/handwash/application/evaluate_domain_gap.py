"""Caso de uso: teste de domain gap (Dia 3) - avalia o modelo ja treinado (Keras +
TFLite, ambos do Dia 2, nenhum retreino aqui) contra video(s) proprios gravados com
camera de celular, o mais perto possivel do setup real dos oculos.

Reusa a MESMA logica de extracao/janelamento/normalizacao dos scripts 03/04, sem
reimplementar nada (CLAUDE.md: "sem inventar logica nova"):
- extract_video_landmarks() + interpolate_short_gaps() = a mesma extracao silver do
  script 03 (infrastructure.landmark_extractor, domain.gap_filling), so que escrevendo
  em CUSTOM_LANDMARKS_DIR em vez de SILVER_LANDMARKS_DIR.
- build_video_features(landmarks_dir=CUSTOM_LANDMARKS_DIR) = a mesma reamostragem
  15fps + normalize_and_mask() do script 04 (application.build_windows), parametrizada
  pra ler de um diretorio diferente.
- window_start_indices()/slice_windows() com o stride de val/teste (1.0s, nao o de
  treino) - isso e uma avaliacao, nao gera dado de treino.

Cada video do manifest custom e o equivalente de UM video Kaggle: um unico passo OMS
do inicio ao fim, rotulo no nivel do video - mesma convencao do dataset original, pra
nao precisar inventar anotacao temporal por segmento (fora de escopo aqui).

Reporta separado: taxa de deteccao do MediaPipe por video (diagnostico "a camera do
celular deixa a mao visivel?") e a acuracia do classificador em cima disso
(diagnostico "dado o que foi visto, o modelo classifica certo?") - importante nao
confundir os dois efeitos, ver DIA1_HANDOFF.md Secao 3, item 4 sobre essa mesma
distincao no dataset Kaggle.
"""
import csv
import json

import numpy as np
import tensorflow as tf

from handwash.application.build_windows import build_video_features
from handwash.application.export_tflite import CHECKPOINT_PATH, TFLITE_PATH
from handwash.config.constants import CLASSES, TARGET_FPS, VAL_TEST_STRIDE_SEC, WINDOW_STEPS
from handwash.config.paths import CHECKPOINTS_DIR, CUSTOM_LANDMARKS_DIR, CUSTOM_MANIFEST, CUSTOM_RAW_DIR
from handwash.domain.class_scheme import class_to_index
from handwash.domain.gap_filling import at_least_one_hand_rate, interpolate_short_gaps
from handwash.domain.windowing import slice_windows, window_start_indices
from handwash.infrastructure.landmark_extractor import extract_video_landmarks

DOMAIN_GAP_REPORT_PATH = CHECKPOINTS_DIR / "domain_gap_report.json"
_EVAL_STRIDE_STEPS = round(VAL_TEST_STRIDE_SEC * TARGET_FPS)

# Numeros de referencia do Dia 2 (dataset Kaggle, camera fixa de pia) - pra comparacao
# honesta, nao pra forcar o resultado do celular a parecer melhor do que e.
KAGGLE_TEST_ACCURACY_KERAS = 0.9719
KAGGLE_TEST_ACCURACY_TFLITE = 0.9610


def _read_custom_manifest() -> list[dict]:
    if not CUSTOM_MANIFEST.exists():
        raise FileNotFoundError(
            f"{CUSTOM_MANIFEST} nao existe. Crie o CSV com colunas "
            f"'video_stem,filename,class_name' (filename relativo a {CUSTOM_RAW_DIR}) "
            "antes de rodar o teste de domain gap."
        )
    with CUSTOM_MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{CUSTOM_MANIFEST} existe mas esta vazio.")
    for row in rows:
        if row["class_name"] not in CLASSES:
            raise ValueError(
                f"class_name {row['class_name']!r} (video_stem={row['video_stem']}) nao esta "
                f"em CLASSES: {CLASSES}"
            )
    return rows


def _extract_custom_video(row: dict, model_path) -> dict:
    """Mesma logica de _process_one_video() do script 03 (extract_landmarks.py),
    aplicada a um video fora do dataset Kaggle. skip_existing implicito: se o .npz ja
    existe, reusa em vez de reprocessar (mesmo padrao de custo do script 03)."""
    video_stem = row["video_stem"]
    npz_path = CUSTOM_LANDMARKS_DIR / f"{video_stem}.npz"

    if npz_path.exists():
        with np.load(npz_path) as cached:
            presence = cached["presence"]
    else:
        video_path = CUSTOM_RAW_DIR / row["filename"]
        if not video_path.exists():
            raise FileNotFoundError(f"Video nao encontrado: {video_path}")
        raw = extract_video_landmarks(video_path, model_path=model_path)

        hand_lm, presence = interpolate_short_gaps(raw.hand_landmarks, raw.presence)
        world_lm, _ = interpolate_short_gaps(raw.world_landmarks, raw.presence)

        CUSTOM_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            hand_landmarks=hand_lm,
            world_landmarks=world_lm,
            presence=presence,
            timestamps_sec=raw.timestamps_sec,
        )

    return {
        "video_stem": video_stem,
        "at_least_one_hand_rate": round(at_least_one_hand_rate(presence), 4),
        "n_frames": int(presence.shape[0]),
    }


def _windows_for_custom_video(row: dict) -> np.ndarray:
    class_idx = class_to_index(row["class_name"])
    features, _ = build_video_features(row["video_stem"], normalize=True, landmarks_dir=CUSTOM_LANDMARKS_DIR)
    starts = window_start_indices(features.shape[0], WINDOW_STEPS, _EVAL_STRIDE_STEPS)
    windows = slice_windows(features, starts, WINDOW_STEPS)
    return windows, class_idx


def _run_tflite(raw_windows: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    probs = np.zeros((len(raw_windows), len(CLASSES)), dtype=np.float32)
    for i, window in enumerate(raw_windows):
        interpreter.set_tensor(in_detail["index"], window[np.newaxis, ...].astype(in_detail["dtype"]))
        interpreter.invoke()
        probs[i] = interpreter.get_tensor(out_detail["index"])[0]
    return probs


def _score(preds: np.ndarray, y: np.ndarray) -> dict:
    overall = float(np.mean(preds == y)) if len(y) else None
    per_class = {}
    for class_idx, class_name in enumerate(CLASSES):
        mask = y == class_idx
        if mask.sum() > 0:
            per_class[class_name] = float(np.mean(preds[mask] == y[mask]))
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for true_idx, pred_idx in zip(y, preds):
        confusion[true_idx, pred_idx] += 1
    return {
        "accuracy_overall": overall,
        "accuracy_per_class": per_class,
        "confusion_matrix": {"classes": CLASSES, "matrix": confusion.tolist()},
    }


def evaluate_domain_gap(model_path=None) -> dict:
    from handwash.config.paths import HAND_LANDMARKER_TASK

    if model_path is None:
        model_path = HAND_LANDMARKER_TASK

    manifest_rows = _read_custom_manifest()

    detection_diagnostics = []
    all_windows_normalized = []
    all_windows_raw = []
    all_y = []
    per_video_true = {}
    per_video_window_slice = {}
    cursor = 0

    for row in manifest_rows:
        detection_diagnostics.append(_extract_custom_video(row, model_path))

        windows_norm, class_idx = _windows_for_custom_video(row)
        windows_raw, _ = build_video_features(
            row["video_stem"], normalize=False, landmarks_dir=CUSTOM_LANDMARKS_DIR
        )
        starts = window_start_indices(windows_raw.shape[0], WINDOW_STEPS, _EVAL_STRIDE_STEPS)
        windows_raw = slice_windows(windows_raw, starts, WINDOW_STEPS)

        n = windows_norm.shape[0]
        if n == 0:
            per_video_window_slice[row["video_stem"]] = (cursor, cursor)
            per_video_true[row["video_stem"]] = class_idx
            continue

        all_windows_normalized.append(windows_norm)
        all_windows_raw.append(windows_raw)
        all_y.extend([class_idx] * n)
        per_video_window_slice[row["video_stem"]] = (cursor, cursor + n)
        per_video_true[row["video_stem"]] = class_idx
        cursor += n

    n_videos_zero_windows = [vs for vs, (a, b) in per_video_window_slice.items() if a == b]

    y = np.array(all_y, dtype=np.int64)
    X_norm = (
        np.concatenate(all_windows_normalized, axis=0)
        if all_windows_normalized
        else np.zeros((0, WINDOW_STEPS, 128), dtype=np.float32)
    )
    X_raw = (
        np.concatenate(all_windows_raw, axis=0)
        if all_windows_raw
        else np.zeros((0, WINDOW_STEPS, 128), dtype=np.float32)
    )

    report = {
        "n_custom_videos": len(manifest_rows),
        "n_total_windows": int(len(y)),
        "videos_with_zero_windows": n_videos_zero_windows,
        "detection_diagnostics": detection_diagnostics,
        "kaggle_reference": {
            "test_accuracy_keras": KAGGLE_TEST_ACCURACY_KERAS,
            "test_accuracy_tflite": KAGGLE_TEST_ACCURACY_TFLITE,
        },
    }

    if len(y) == 0:
        report["warning"] = (
            "Nenhuma janela produzida a partir dos videos custom (todos mais curtos que "
            "2.0s apos reamostragem, ou todos falharam na extracao) - nada pra avaliar."
        )
        DOMAIN_GAP_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(DOMAIN_GAP_REPORT_PATH)
        return report

    keras_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    keras_probs = keras_model.predict(X_norm, verbose=0)
    keras_preds = np.argmax(keras_probs, axis=1)
    keras_confidences = np.max(keras_probs, axis=1)

    tflite_probs = _run_tflite(X_raw)
    tflite_preds = np.argmax(tflite_probs, axis=1)
    tflite_confidences = np.max(tflite_probs, axis=1)

    keras_score = _score(keras_preds, y)
    tflite_score = _score(tflite_preds, y)

    keras_correct = keras_preds == y
    tflite_correct = tflite_preds == y

    def _majority_vote(preds: np.ndarray) -> dict:
        n_correct = 0
        per_video = {}
        for video_stem, (a, b) in per_video_window_slice.items():
            true_idx = per_video_true[video_stem]
            if a == b:
                per_video[video_stem] = {"true": CLASSES[true_idx], "predicted": None, "correct": None}
                continue
            votes = preds[a:b]
            vote_counts = np.bincount(votes, minlength=len(CLASSES))
            majority_pred = int(np.argmax(vote_counts))
            correct = majority_pred == true_idx
            n_correct += int(correct)
            per_video[video_stem] = {
                "true": CLASSES[true_idx],
                "predicted": CLASSES[majority_pred],
                "correct": bool(correct),
            }
        n_scored = sum(1 for a, b in per_video_window_slice.values() if a != b)
        return {
            "per_video_accuracy_majority_vote": n_correct / n_scored if n_scored else None,
            "per_video_detail": per_video,
        }

    report.update(
        {
            "keras": {
                **keras_score,
                "mean_confidence_correct": (
                    float(keras_confidences[keras_correct].mean()) if keras_correct.any() else None
                ),
                "mean_confidence_incorrect": (
                    float(keras_confidences[~keras_correct].mean()) if (~keras_correct).any() else None
                ),
                "accuracy_drop_vs_kaggle_pp": (
                    (KAGGLE_TEST_ACCURACY_KERAS - keras_score["accuracy_overall"]) * 100
                    if keras_score["accuracy_overall"] is not None
                    else None
                ),
                **_majority_vote(keras_preds),
            },
            "tflite": {
                **tflite_score,
                "mean_confidence_correct": (
                    float(tflite_confidences[tflite_correct].mean()) if tflite_correct.any() else None
                ),
                "mean_confidence_incorrect": (
                    float(tflite_confidences[~tflite_correct].mean()) if (~tflite_correct).any() else None
                ),
                "accuracy_drop_vs_kaggle_pp": (
                    (KAGGLE_TEST_ACCURACY_TFLITE - tflite_score["accuracy_overall"]) * 100
                    if tflite_score["accuracy_overall"] is not None
                    else None
                ),
                **_majority_vote(tflite_preds),
            },
        }
    )

    DOMAIN_GAP_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(DOMAIN_GAP_REPORT_PATH)
    return report


if __name__ == "__main__":
    summary = evaluate_domain_gap()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
