#!/usr/bin/env python
"""Script de paridade independente - valida handwash_step_classifier.tflite contra
sample_io_pairs.npz sem depender de nada do resto do repositorio handwash_model.

Uso:
    python parity_check.py
    (rodar a partir desta pasta, ou passar os caminhos por argumento - ver abaixo)

Requer: numpy, e um jeito de rodar TFLite. Duas opcoes, na ordem que o script tenta:
1. tensorflow (tf.lite.Interpreter) - se o time Android/ML ja tem TF instalado.
2. tflite_runtime (pacote leve, sem o resto do TensorFlow) - se so isso estiver disponivel.

O que este script confirma:
- O .tflite carrega e aceita entrada (1, 30, 128) float32.
- A saida bate (dentro de uma tolerancia pequena - INT8 nao e bit-exato) com o que foi
  gravado em sample_io_pairs.npz no momento da exportacao.
- A classe prevista (argmax) bate com a classe verdadeira nas 3 amostras.

Se este script passar, o .tflite esta se comportando como documentado no MODEL_CARD.md.
Se falhar, NAO adapte silenciosamente a tolerancia so pra passar - o objetivo e pegar
divergencia real entre o ambiente de export e o ambiente de execucao (versao de TFLite,
plataforma, etc.), nao esconder ela.
"""
import json
import sys
from pathlib import Path

import numpy as np

TOLERANCE = 0.05  # INT8 nao e bit-exato entre plataformas - tolerancia generosa de proposito


def _load_interpreter(tflite_path: str):
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=tflite_path)
    except ImportError:
        pass
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=tflite_path)
    except ImportError:
        raise ImportError(
            "Nem 'tensorflow' nem 'tflite_runtime' encontrados. Instale um dos dois: "
            "'pip install tensorflow' ou 'pip install tflite-runtime'."
        )


def main(artifacts_dir: str = ".") -> int:
    artifacts_dir = Path(artifacts_dir)
    tflite_path = artifacts_dir / "handwash_step_classifier.tflite"
    sample_io_path = artifacts_dir / "sample_io_pairs.npz"
    classes_path = artifacts_dir / "classes.json"

    classes = json.loads(classes_path.read_text(encoding="utf-8"))

    with np.load(sample_io_path) as data:
        input_windows = data["input_windows"]  # (N, 30, 128) float32, landmarks BRUTOS
        expected_output_probs = data["expected_output_probs"]  # (N, 7) float32
        true_class_idx = data["true_class_idx"]  # (N,) int64

    interpreter = _load_interpreter(str(tflite_path))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    print(f"Input esperado pelo modelo: shape={in_detail['shape']}, dtype={in_detail['dtype']}")
    print(f"Output do modelo: shape={out_detail['shape']}, dtype={out_detail['dtype']}")
    print()

    all_ok = True
    for i, (window, expected_probs, true_idx) in enumerate(
        zip(input_windows, expected_output_probs, true_class_idx)
    ):
        interpreter.set_tensor(in_detail["index"], window[np.newaxis, ...].astype(in_detail["dtype"]))
        interpreter.invoke()
        actual_probs = interpreter.get_tensor(out_detail["index"])[0]

        max_diff = float(np.max(np.abs(actual_probs - expected_probs)))
        predicted_idx = int(np.argmax(actual_probs))
        pred_matches_true = predicted_idx == int(true_idx)
        within_tolerance = max_diff <= TOLERANCE

        status = "OK" if (pred_matches_true and within_tolerance) else "FALHOU"
        print(
            f"[{status}] amostra {i} ({classes[true_idx]}): "
            f"previsto={classes[predicted_idx]}, diff_max={max_diff:.4f}"
        )
        all_ok = all_ok and pred_matches_true and within_tolerance

    print()
    print("RESULTADO: TUDO OK" if all_ok else "RESULTADO: HA DIVERGENCIA - ver MODEL_CARD.md")
    return 0 if all_ok else 1


if __name__ == "__main__":
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    sys.exit(main(artifacts_dir))
