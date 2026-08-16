"""Caso de uso: checagem de viabilidade "TFLite builtin-ops only" (Secao 7.1 do plano -
GATE, rodar antes do treino real).

Monta um modelo-placeholder com os MESMOS tipos de camada da arquitetura principal
(Conv1D+GRU+Dense) e tenta converter pra TFLite sem SELECT_TF_OPS/Flex. Builtin-only ja
e o padrao do conversor - "converteu e rodou sem erro" confirma viabilidade sem custar
nem um segundo de treino de verdade. Pesos sao aleatorios/nao treinados de proposito: o
teste e sobre COMPATIBILIDADE DE OPERACAO, nunca sobre acuracia.

Ordem de fallback (Secao 7.1 do plano, na ordem documentada - nao pular etapa):
1. Opcao A como esta (Conv1D+LayerNorm+Conv1D+GRU+Dense, ~89K params).
2. Opcao A com GRU(unroll=True) - GRU e o suspeito mais provavel de operacao nao
   suportada (confirmado: o erro real e "tf.TensorListReserve ... requires
   element_shape to be static", do laco dinamico do GRU).
3. Opcao C (so convolucional, zero risco de op) - fallback definitivo.
SELECT_TF_OPS/Flex NUNCA e tentado aqui - e ultimo recurso e esta fora do escopo deste
teste (o proprio plano nota que foi o que forcou o edi-riga a uma dependencia mais
pesada no lado Android, exatamente o que este projeto quer evitar).
"""
import numpy as np

from handwash.config.constants import CLASSES, WINDOW_FEATURE_DIM, WINDOW_STEPS
from handwash.infrastructure.model_factory import build_fallback_c_model, build_primary_model
from handwash.infrastructure.tflite_converter import convert_to_tflite_builtin_only, run_tflite_inference

_RANDOM_SEED = 42  # reprodutibilidade do dado placeholder - nao afeta o resultado do teste


def _random_input() -> np.ndarray:
    rng = np.random.default_rng(_RANDOM_SEED)
    return rng.standard_normal((1, WINDOW_STEPS, WINDOW_FEATURE_DIM)).astype(np.float32)


def _attempt(label: str, build_fn, **build_kwargs) -> dict:
    model = build_fn(**build_kwargs)
    n_params = model.count_params()

    try:
        tflite_bytes = convert_to_tflite_builtin_only(model)
    except Exception as exc:
        return {
            "label": label, "success": False, "stage": "convert", "n_params": n_params,
            "tflite_size_bytes": None, "output_shape": None, "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        output = run_tflite_inference(tflite_bytes, _random_input())
    except Exception as exc:
        return {
            "label": label, "success": False, "stage": "run", "n_params": n_params,
            "tflite_size_bytes": len(tflite_bytes), "output_shape": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    expected_shape = (1, len(CLASSES))
    if output.shape != expected_shape or not np.all(np.isfinite(output)):
        return {
            "label": label, "success": False, "stage": "output_check", "n_params": n_params,
            "tflite_size_bytes": len(tflite_bytes), "output_shape": list(output.shape),
            "error": f"shape/finitude inesperada: shape={output.shape}, esperado={expected_shape}",
        }

    return {
        "label": label, "success": True, "stage": None, "n_params": n_params,
        "tflite_size_bytes": len(tflite_bytes), "output_shape": list(output.shape), "error": None,
    }


def check_builtin_ops_viability() -> dict:
    attempts = [_attempt("opcao_a_como_esta", build_primary_model)]

    if not attempts[-1]["success"]:
        attempts.append(_attempt("opcao_a_gru_unroll_true", build_primary_model, gru_unroll=True))

    if not attempts[-1]["success"]:
        attempts.append(_attempt("opcao_c_conv_only", build_fallback_c_model))

    winner = next((a for a in attempts if a["success"]), None)
    return {
        "gate_passed": winner is not None,
        "winning_architecture": winner["label"] if winner else None,
        "winning_n_params": winner["n_params"] if winner else None,
        "winning_tflite_size_bytes": winner["tflite_size_bytes"] if winner else None,
        "attempts": attempts,
    }


if __name__ == "__main__":
    summary = check_builtin_ops_viability()
    print(summary)
