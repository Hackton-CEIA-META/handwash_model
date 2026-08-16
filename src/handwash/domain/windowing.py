"""Fatiamento de uma sequencia de features por video em janelas fixas (Secao 4 do plano).

Janela = WINDOW_STEPS passos (2.0s a 15fps = 30). Janela incompleta no final da sequencia
e DESCARTADA, nunca preenchida com padding - o contrato de buffer do runtime so roda
inferencia depois que o buffer de 2.0s esta cheio (nunca uma janela parcial), entao
treinar com janelas padded ensinaria o modelo a esperar uma entrada que ele nunca ve em
producao.
"""
import numpy as np


def window_start_indices(n_frames: int, window_steps: int, stride_steps: int) -> list[int]:
    """Indices de frame onde cada janela comeca. Lista vazia se n_frames < window_steps
    (sequencia mais curta que uma janela inteira) - o range() ja resolve isso sozinho,
    sem precisar de um caso especial."""
    last_start = n_frames - window_steps
    return list(range(0, last_start + 1, stride_steps))


def slice_windows(features: np.ndarray, starts: list[int], window_steps: int) -> np.ndarray:
    """features: array[n_frames, F]. Retorna array[n_windows, window_steps, F]."""
    if not starts:
        return np.zeros((0, window_steps) + features.shape[1:], dtype=features.dtype)
    return np.stack([features[s:s + window_steps] for s in starts])
