"""Caso de uso: montar o pacote de handoff pro time Android (Secao 8 do plano).

Junta em models/artifacts/: o .tflite (ja exportado por application.export_tflite),
classes.json (copiado do gold, congelado), e sample_io_pairs.npz (janelas reais de
teste + saida esperada do .tflite, pro time Android escrever um teste unitario sem
precisar rodar Python/TF). MODEL_CARD.md e parity_check.py sao documentos/scripts
autorais, nao gerados aqui.
"""
import json
import shutil

import numpy as np
import tensorflow as tf

from handwash.application.evaluate_tflite_vs_keras import _raw_test_windows_aligned_to_gold
from handwash.application.export_tflite import TFLITE_PATH
from handwash.config.constants import CLASSES
from handwash.config.paths import ARTIFACTS_DIR, GOLD_CLASSES_JSON

SAMPLE_IO_PATH = ARTIFACTS_DIR / "sample_io_pairs.npz"
ARTIFACTS_CLASSES_JSON = ARTIFACTS_DIR / "classes.json"
N_SAMPLE_PAIRS = 3


def _run_tflite_single(interpreter, window: np.ndarray) -> np.ndarray:
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(in_detail["index"], window[np.newaxis, ...].astype(in_detail["dtype"]))
    interpreter.invoke()
    return interpreter.get_tensor(out_detail["index"])[0]


def build_handoff_package() -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(GOLD_CLASSES_JSON, ARTIFACTS_CLASSES_JSON)

    raw_X_test, y_test = _raw_test_windows_aligned_to_gold()
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    rng = np.random.default_rng(7)
    # escolhe N_SAMPLE_PAIRS janelas de classes DIFERENTES (nao repetidas), pra cobrir
    # variedade em vez de N amostras aleatorias que podem cair todas na mesma classe
    chosen_idx = []
    seen_classes = set()
    for idx in rng.permutation(len(y_test)):
        class_idx = int(y_test[idx])
        if class_idx not in seen_classes:
            chosen_idx.append(int(idx))
            seen_classes.add(class_idx)
        if len(chosen_idx) >= N_SAMPLE_PAIRS:
            break

    sample_inputs = raw_X_test[chosen_idx]
    sample_true_labels = y_test[chosen_idx]
    sample_outputs = np.stack([_run_tflite_single(interpreter, w) for w in sample_inputs])

    np.savez_compressed(
        SAMPLE_IO_PATH,
        input_windows=sample_inputs.astype(np.float32),
        expected_output_probs=sample_outputs.astype(np.float32),
        true_class_idx=sample_true_labels.astype(np.int64),
        classes=np.array(CLASSES),
    )

    return {
        "artifacts_dir": str(ARTIFACTS_DIR),
        "classes_json": str(ARTIFACTS_CLASSES_JSON),
        "sample_io_pairs": str(SAMPLE_IO_PATH),
        "n_sample_pairs": len(chosen_idx),
        "sample_classes": [CLASSES[i] for i in sample_true_labels],
    }


if __name__ == "__main__":
    summary = build_handoff_package()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
