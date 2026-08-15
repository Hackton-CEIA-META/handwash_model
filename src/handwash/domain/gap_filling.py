"""Preenchimento de gaps de deteccao (Secao 4.3 do plano) - logica pura sobre arrays ja extraidos.

Gaps curtos (<= MAX_SHORT_GAP_FRAMES) entre duas deteccoes validas sao interpolados
linearmente. Gaps longos ou nas pontas do video continuam zero-preenchidos, com
presence=0 - nao mentimos pro modelo dizendo que houve deteccao onde nao houve.
"""
import numpy as np

MAX_SHORT_GAP_FRAMES = 5


def interpolate_short_gaps(landmarks: np.ndarray, presence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    landmarks: float32[n_frames, n_slots, 21, 3]
    presence: uint8[n_frames, n_slots]
    """
    landmarks = landmarks.copy()
    presence = presence.copy()
    n_frames, n_slots = presence.shape

    for slot in range(n_slots):
        valid_frames = np.where(presence[:, slot] == 1)[0]
        if len(valid_frames) < 2:
            continue

        for i in range(len(valid_frames) - 1):
            start, end = int(valid_frames[i]), int(valid_frames[i + 1])
            gap = end - start
            if 1 < gap <= MAX_SHORT_GAP_FRAMES + 1:
                for t in range(start + 1, end):
                    alpha = (t - start) / gap
                    landmarks[t, slot] = (
                        (1 - alpha) * landmarks[start, slot] + alpha * landmarks[end, slot]
                    )
                    presence[t, slot] = 1

    return landmarks, presence


def detection_rate(presence: np.ndarray) -> float:
    """Fracao de (frame, slot) com presence=1, tratando os 2 slots como unidades
    independentes. CUIDADO: varios passos da tecnica OMS sobrepoem/entrelacam as
    maos por definicao - isso derruba essa metrica mesmo com deteccao saudavel.
    Para avaliar viabilidade de fato, prefira at_least_one_hand_rate() (Secao 1.3
    do plano) e trate both_hands_rate() como informativo, nao como gate."""
    return float(presence.mean())


def at_least_one_hand_rate(presence: np.ndarray) -> float:
    """Fracao de frames com PELO MENOS uma mao detectada. E a metrica que mais
    importa pra viabilidade: se nenhuma mao e vista, nao ha sinal nenhum pro
    modelo naquele frame. Quando so uma mao e vista, o presence flag da outra
    fica 0 e o modelo aprende a nao confiar naquele slot - nao e uma falha."""
    return float(presence.any(axis=1).mean())


def both_hands_rate(presence: np.ndarray) -> float:
    """Fracao de frames com as DUAS maos detectadas simultaneamente. Informativo,
    nao e o criterio de gate - passos com muita sobreposicao de maos (ex.: dedos
    entrelacados) legitimamente tem essa taxa mais baixa."""
    return float(presence.all(axis=1).mean())
