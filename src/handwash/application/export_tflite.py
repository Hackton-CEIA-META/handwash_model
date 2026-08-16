"""Caso de uso: exportacao TFLite INT8 full-integer (Secao 7 do plano).

Compoe [NormalizeAndMaskLayer, sem pesos] + [modelo treinado, com pesos reais] num unico
grafo - a normalizacao fica embutida no .tflite exportado, entao o CONTRATO DE ENTRADA
MUDA em relacao ao treino: o modelo exportado recebe landmarks BRUTOS (reamostrados a
15fps/30 passos, mas NAO normalizados), nao os arrays ja normalizados de
data/gold/*/windows.npz. O dataset representativo pra calibracao INT8 por isso tambem
precisa ser bruto, amostrado SO do split de treino (nunca val/teste - Secao 7.2 do
plano), reconstruido a partir do silver com o mesmo stride de treino
(build_windows.build_video_features(normalize=False)).

Quantizacao full-integer (pesos + ativacoes INT8), mantendo entrada/saida em float32 -
nao setar inference_input_type/inference_output_type e o que garante isso (Secao 7.2 do
plano: "simplifica o contrato com o Android... sem matematica de escala/zero-point do
lado Kotlin").
"""
import csv

import numpy as np
import tensorflow as tf

from handwash.application.build_windows import build_video_features
from handwash.config.constants import TARGET_FPS, TRAIN_STRIDE_SEC, WINDOW_FEATURE_DIM, WINDOW_STEPS
from handwash.config.paths import ARTIFACTS_DIR, BRONZE_MANIFEST, CHECKPOINTS_DIR
from handwash.domain.windowing import slice_windows, window_start_indices
from handwash.infrastructure.normalization_layer import NormalizeAndMaskLayer

CHECKPOINT_PATH = CHECKPOINTS_DIR / "handwash_classifier_best.keras"
TFLITE_PATH = ARTIFACTS_DIR / "handwash_step_classifier.tflite"
N_REPRESENTATIVE_WINDOWS_PER_CLASS = 43  # 7 classes * 43 ~= 300 (Secao 7.2 do plano), estratificado
REPRESENTATIVE_SEED = 42
_TRAIN_STRIDE_STEPS = round(TRAIN_STRIDE_SEC * TARGET_FPS)  # mesma stride do treino (Secao 4 do plano)


def _read_bronze_manifest() -> list[dict]:
    with BRONZE_MANIFEST.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_raw_train_windows(
    n_per_class: int = N_REPRESENTATIVE_WINDOWS_PER_CLASS, seed: int = REPRESENTATIVE_SEED
) -> np.ndarray:
    """Amostra ate n_per_class janelas BRUTAS (reamostradas, sem normalizar) POR CLASSE,
    de videos do split de treino - Secao 7.2 do plano: "so do split de TREINO, nunca
    teste" e "~300 janelas" (7 classes * 43 ~= 300).

    Estratificado por classe, nao proporcional ao tamanho da classe - achado real ao
    investigar a Secao 7.3 do plano ("investigar queda >~2pp, especialmente colapso por
    classe"): uma primeira tentativa amostrou proporcionalmente do pool inteiro (sem
    estratificar) e calibrou mal especificamente step1/step3 (queda de ate 30pp so
    nessas duas - as duas classes menores E mais mutuamente confundidas ja no Keras
    FP32), mesmo depois de corrigir a cobertura de video (ver historico). Estratificar
    garante que toda classe, grande ou pequena, contribui o MESMO numero de exemplos
    pra calibracao - testado empiricamente, ver relatorio de comparacao Keras vs TFLite."""
    train_rows = [row for row in _read_bronze_manifest() if row["split"] == "train"]
    rng = np.random.default_rng(seed)

    pool_by_class: dict[str, list[np.ndarray]] = {}
    for row in train_rows:
        features, _ = build_video_features(row["video_stem"], normalize=False)
        starts = window_start_indices(features.shape[0], WINDOW_STEPS, _TRAIN_STRIDE_STEPS)
        windows = slice_windows(features, starts, WINDOW_STEPS)
        if windows.shape[0] > 0:
            pool_by_class.setdefault(row["merged_class"], []).append(windows)

    sampled = []
    for window_arrays in pool_by_class.values():
        class_pool = np.concatenate(window_arrays, axis=0)
        chosen = rng.choice(class_pool.shape[0], size=min(n_per_class, class_pool.shape[0]), replace=False)
        sampled.append(class_pool[chosen])

    return np.concatenate(sampled, axis=0).astype(np.float32)


def build_export_model() -> tf.keras.Model:
    """Compoe normalizacao (embutida, sem peso) + modelo treinado (pesos reais, reusados
    - nao reinicializados) - o grafo que sera convertido pra TFLite."""
    trained_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    raw_inputs = tf.keras.Input(shape=(WINDOW_STEPS, WINDOW_FEATURE_DIM), name="raw_landmarks_and_presence")
    normalized = NormalizeAndMaskLayer(name="normalize_and_mask")(raw_inputs)
    outputs = trained_model(normalized)
    return tf.keras.Model(raw_inputs, outputs, name="handwash_step_classifier_export")


def export_tflite_int8() -> dict:
    export_model = build_export_model()
    representative_windows = sample_raw_train_windows()

    def representative_dataset():
        for window in representative_windows:
            yield [window[np.newaxis, ...]]  # (1, 30, 128), float32 - Secao 7.2 do plano

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_bytes = converter.convert()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TFLITE_PATH.write_bytes(tflite_bytes)

    return {
        "tflite_path": str(TFLITE_PATH),
        "tflite_size_bytes": len(tflite_bytes),
        "n_representative_windows": int(representative_windows.shape[0]),
    }


if __name__ == "__main__":
    import json
    summary = export_tflite_int8()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
