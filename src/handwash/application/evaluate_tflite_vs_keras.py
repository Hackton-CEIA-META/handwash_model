"""Caso de uso: comparar Keras FP32 vs TFLite INT8 sobre as MESMAS janelas de teste
(Secao 7.3 do plano - "nao assumir que Keras ~= TFLite").

O modelo Keras treinado recebe dado JA normalizado (data/gold/test/windows.npz). O
modelo TFLite exportado normaliza internamente (application.export_tflite), entao
recebe as MESMAS janelas em forma BRUTA - reconstruidas aqui a partir do silver com a
mesma janela/stride que build_windows() usou pro split de teste, alinhadas por
split_row_idx pra garantir que os dois modelos veem exatamente as mesmas 641 janelas,
so que numa representacao diferente cada um.

Investigar (nao so reportar) qualquer queda > ~2pp, especialmente colapso por classe
(Secao 7.3 do plano).
"""
import csv
import json

import numpy as np
import tensorflow as tf

from handwash.application.build_windows import build_video_features
from handwash.application.export_tflite import CHECKPOINT_PATH, TFLITE_PATH
from handwash.config.constants import CLASSES, TARGET_FPS, VAL_TEST_STRIDE_SEC, WINDOW_FEATURE_DIM, WINDOW_STEPS
from handwash.config.paths import CHECKPOINTS_DIR, GOLD_DIR
from handwash.domain.normalization import normalize_and_mask
from handwash.domain.windowing import slice_windows, window_start_indices

COMPARISON_REPORT_PATH = CHECKPOINTS_DIR / "tflite_vs_keras_report.json"
ACCURACY_DROP_GATE_PP = 2.0  # Secao 7.3 do plano: investigar queda > ~2pp
_TEST_STRIDE_STEPS = round(VAL_TEST_STRIDE_SEC * TARGET_FPS)


def _raw_test_windows_aligned_to_gold() -> tuple[np.ndarray, np.ndarray]:
    """Reconstroi as janelas de teste BRUTAS (sem normalizar), na MESMA ordem
    (split_row_idx) de data/gold/test/windows.npz - garante que Keras e TFLite sao
    comparados sobre exatamente as mesmas janelas."""
    with (GOLD_DIR / "manifest.csv").open(newline="", encoding="utf-8") as f:
        test_rows = [row for row in csv.DictReader(f) if row["split"] == "test"]

    by_video: dict[str, list[dict]] = {}
    for row in test_rows:
        by_video.setdefault(row["video_stem"], []).append(row)

    n_test = len(test_rows)
    raw_X = np.zeros((n_test, WINDOW_STEPS, WINDOW_FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n_test, dtype=np.int64)

    for video_stem, rows in by_video.items():
        rows.sort(key=lambda r: int(r["window_idx"]))
        features, _ = build_video_features(video_stem, normalize=False)
        starts = window_start_indices(features.shape[0], WINDOW_STEPS, _TEST_STRIDE_STEPS)
        windows = slice_windows(features, starts, WINDOW_STEPS)
        if windows.shape[0] != len(rows):
            raise ValueError(
                f"Contagem de janela nao bate pra {video_stem}: {windows.shape[0]} reconstruidas "
                f"vs {len(rows)} no gold manifest - alinhamento quebrado"
            )
        for row, window in zip(rows, windows):
            idx = int(row["split_row_idx"])
            raw_X[idx] = window
            y[idx] = int(row["class_idx"])

    return raw_X, y


def _assert_raw_normalizes_back_to_gold(raw_X: np.ndarray, gold_X: np.ndarray, n_samples: int = 20) -> None:
    """Confirma, numa amostra, que normalizar o dado bruto reconstruido reproduz
    exatamente o que esta salvo em data/gold/test/windows.npz - se isso nao bater, o
    alinhamento entre as duas representacoes esta quebrado e a comparacao Keras/TFLite
    abaixo nao seria sobre as mesmas janelas de verdade."""
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(raw_X), size=min(n_samples, len(raw_X)), replace=False)
    for idx in sample_idx:
        landmarks = raw_X[idx, :, :126].reshape(WINDOW_STEPS, 2, 21, 3)
        presence = raw_X[idx, :, 126:]
        renormalized = normalize_and_mask(landmarks, presence).reshape(WINDOW_STEPS, 126)
        renormalized_full = np.concatenate([renormalized, presence], axis=-1)
        if not np.allclose(renormalized_full, gold_X[idx], atol=1e-4):
            raise ValueError(f"Janela {idx}: bruto renormalizado nao bate com o gold - alinhamento quebrado")


def _run_tflite(raw_X: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    probs = np.zeros((len(raw_X), len(CLASSES)), dtype=np.float32)
    for i, window in enumerate(raw_X):
        interpreter.set_tensor(in_detail["index"], window[np.newaxis, ...].astype(in_detail["dtype"]))
        interpreter.invoke()
        probs[i] = interpreter.get_tensor(out_detail["index"])[0]
    return probs


def evaluate_tflite_vs_keras() -> dict:
    with np.load(GOLD_DIR / "test" / "windows.npz") as d:
        X_test_normalized, y_test = d["X"], d["y"]

    raw_X_test, y_test_raw = _raw_test_windows_aligned_to_gold()
    if not np.array_equal(y_test, y_test_raw):
        raise ValueError("Rotulos do gold e da reconstrucao bruta nao batem - alinhamento quebrado")
    _assert_raw_normalizes_back_to_gold(raw_X_test, X_test_normalized)

    keras_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    keras_probs = keras_model.predict(X_test_normalized, verbose=0)
    keras_preds = np.argmax(keras_probs, axis=1)

    tflite_probs = _run_tflite(raw_X_test)
    tflite_preds = np.argmax(tflite_probs, axis=1)

    keras_accuracy = float(np.mean(keras_preds == y_test))
    tflite_accuracy = float(np.mean(tflite_preds == y_test))
    drop_pp = (keras_accuracy - tflite_accuracy) * 100

    per_class = {}
    for class_idx, class_name in enumerate(CLASSES):
        mask = y_test == class_idx
        if mask.sum() > 0:
            per_class[class_name] = {
                "keras": float(np.mean(keras_preds[mask] == y_test[mask])),
                "tflite": float(np.mean(tflite_preds[mask] == y_test[mask])),
                "drop_percentage_points": float(
                    (np.mean(keras_preds[mask] == y_test[mask]) - np.mean(tflite_preds[mask] == y_test[mask])) * 100
                ),
            }

    report = {
        "keras_accuracy": keras_accuracy,
        "tflite_accuracy": tflite_accuracy,
        "accuracy_drop_percentage_points": drop_pp,
        "gate_2pp_passed": drop_pp <= ACCURACY_DROP_GATE_PP,
        "keras_tflite_agreement_rate": float(np.mean(keras_preds == tflite_preds)),
        "per_class": per_class,
        "n_test_windows": int(len(y_test)),
    }
    COMPARISON_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(COMPARISON_REPORT_PATH)
    return report


if __name__ == "__main__":
    summary = evaluate_tflite_vs_keras()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
