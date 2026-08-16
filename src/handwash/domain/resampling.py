"""Reamostragem de landmarks por timestamp: 30fps nativo -> 15fps alvo (Secao 4 do plano).

Constroi uma grade de tempo REGULAR a partir da duracao real do video e interpola sobre
os timestamps nativos - nunca pula frame por indice. O fps nativo varia levemente por
video (confirmado no Dia 1: 30.04-30.23), e o fps entregue pelo DAT via Bluetooth em
runtime tambem nao e um metronomo perfeito, entao o pipeline de treino precisa ser
robusto a isso desde o inicio, nao so "funcionar" no caso feliz de fps exato.

Landmarks (coordenadas continuas) usam interpolacao LINEAR entre os dois frames nativos
vizinhos. O flag `presence` (binario) usa o frame nativo mais PROXIMO (nearest) em vez de
interpolar - misturar 0/1 linearmente produziria um valor fracionario sem sentido pra um
flag de deteccao.
"""
import numpy as np


def target_grid(duration_sec: float, target_fps: float) -> np.ndarray:
    """Grade regular de timestamps [0, 1/fps, 2/fps, ...] cobrindo a duracao do video."""
    n_target = max(1, int(np.floor(duration_sec * target_fps)) + 1)
    return np.arange(n_target, dtype=np.float32) / np.float32(target_fps)


def _bracket_indices(timestamps_sec: np.ndarray, target_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Para cada timestamp alvo, acha o par de frames nativos (idx_lo, idx_hi) que o
    cerca e a fracao alpha entre eles. Fora do intervalo nativo, satura no frame de
    borda (idx_lo == idx_hi, alpha 0 ou 1) - nunca extrapola."""
    n_source = len(timestamps_sec)
    idx_hi = np.searchsorted(timestamps_sec, target_ts, side="left").clip(1, n_source - 1)
    idx_lo = idx_hi - 1
    ts_lo, ts_hi = timestamps_sec[idx_lo], timestamps_sec[idx_hi]
    span = np.where(ts_hi > ts_lo, ts_hi - ts_lo, 1.0)
    alpha = np.clip((target_ts - ts_lo) / span, 0.0, 1.0)
    return idx_lo, idx_hi, alpha


def resample_linear(values: np.ndarray, timestamps_sec: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """values: array[n_source_frames, ...] - qualquer shape de cauda (ex.: landmarks)."""
    idx_lo, idx_hi, alpha = _bracket_indices(timestamps_sec, target_ts)
    alpha = alpha.reshape((-1,) + (1,) * (values.ndim - 1))
    blended = (1 - alpha) * values[idx_lo] + alpha * values[idx_hi]
    return blended.astype(values.dtype)


def resample_nearest(values: np.ndarray, timestamps_sec: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """values: array[n_source_frames, ...] - usa o frame nativo mais proximo, sem misturar."""
    idx_lo, idx_hi, alpha = _bracket_indices(timestamps_sec, target_ts)
    nearest = np.where(alpha < 0.5, idx_lo, idx_hi)
    return values[nearest]
