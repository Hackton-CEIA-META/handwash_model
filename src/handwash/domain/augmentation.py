"""Augmentacao de janelas (Secao 5, item 5 do plano) - aplicada SO no treino, nunca em
val/teste. Cada funcao opera sobre uma janela ja normalizada (30, 128) = 126 coords de
landmark (2 maos x 21 pontos x 3D, saida de domain.normalization.normalize_and_mask) + 2
flags de presenca, o mesmo layout que sai do gold layer (application.build_windows).

Todas preservam o contrato presence=0 => landmarks exatamente zero (a mesma garantia que
normalize_and_mask estabelece no gold layer). mirror_lr, rotate_z e scale_jitter sao
seguras por construcao (transformacoes lineares que preservam a origem, entao um vetor
zero exato continua zero exato). jitter_time e landmark_noise PODEM quebrar isso (a
primeira mistura frames vizinhos que podem ter presence diferente; a segunda soma ruido
em cima de zero) - as duas remascaram por presenca depois de transformar.

Nunca embaralha a ordem dos passos - e sinal temporal real (Secao 5 do plano).

`jitter_time` e uma aproximacao deliberada do que o plano descreve ("+/-1 frame nativo
antes de reamostrar"): o gold layer so persiste a janela ja reamostrada a 15fps, nao a
sequencia intermediaria por video antes de fatiar em janelas, entao um jitter fiel ao pe
da letra exigiria reprocessar do silver bruto a cada epoca - caro demais pra uma
augmentation pensada pra ser leve. Aqui o deslocamento e aplicado direto sobre a janela
final, reusando a mesma interpolacao do gold layer (domain.resampling).
"""
import numpy as np

from handwash.config.constants import NUM_LANDMARK_COORDS, TARGET_FPS
from handwash.domain.resampling import resample_linear, resample_nearest

_NUM_SLOTS = 2
_NUM_LANDMARKS = 21
_NUM_COORDS = 3


def _split(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """window: array[T, 128] -> (landmarks[T, 2, 21, 3], presence[T, 2])."""
    landmarks_flat = window[:, :NUM_LANDMARK_COORDS]
    presence = window[:, NUM_LANDMARK_COORDS:]
    landmarks = landmarks_flat.reshape(window.shape[0], _NUM_SLOTS, _NUM_LANDMARKS, _NUM_COORDS)
    return landmarks, presence


def _join(landmarks: np.ndarray, presence: np.ndarray) -> np.ndarray:
    flat = landmarks.reshape(landmarks.shape[0], NUM_LANDMARK_COORDS)
    return np.concatenate([flat, presence], axis=-1).astype(np.float32)


def mirror_lr(window: np.ndarray) -> np.ndarray:
    """Espelha horizontalmente (flip x) e troca os slots esquerda<->direita - com o
    esquema de 7 classes ja mesclando L/R (Secao 2 do plano), preserva o rotulo
    automaticamente, sem precisar trocar rotulo nenhum."""
    landmarks, presence = _split(window)
    mirrored_landmarks = landmarks[:, ::-1, :, :].copy()  # troca slot 0<->1
    mirrored_landmarks[..., 0] *= -1  # flip x
    mirrored_presence = presence[:, ::-1].copy()
    return _join(mirrored_landmarks, mirrored_presence)


def rotate_z(window: np.ndarray, max_degrees: float, rng: np.random.Generator) -> np.ndarray:
    """Rotaciona em torno do eixo Z por um angulo aleatorio em [-max_degrees, max_degrees]
    (Secao 5 do plano: +/-10-15 graus, simula angulo de camera fixa vs. egocentrica) - UM
    angulo por janela inteira (nao por frame): uma diferenca de angulo de camera e
    estrutural, persiste durante o clipe todo, nao muda frame a frame. Segura por
    construcao pra frames ausentes - rotacao e linear e preserva a origem (0 rotacionado
    continua 0), sem precisar remascarar."""
    landmarks, presence = _split(window)
    theta = np.deg2rad(rng.uniform(-max_degrees, max_degrees))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x, y, z = landmarks[..., 0], landmarks[..., 1], landmarks[..., 2]
    rotated = np.stack([x * cos_t - y * sin_t, x * sin_t + y * cos_t, z], axis=-1)
    return _join(rotated, presence)


def scale_jitter(window: np.ndarray, scale_range: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Escala uniformemente por um fator aleatorio em scale_range (Secao 5 do plano:
    [0.9, 1.1]) - UM fator por janela, mesmo raciocinio de rotate_z. Segura por
    construcao pra frames ausentes (0 * fator = 0)."""
    landmarks, presence = _split(window)
    factor = rng.uniform(scale_range[0], scale_range[1])
    return _join(landmarks * factor, presence)


def landmark_noise(window: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano pequeno e independente por (frame, mao, ponto, coordenada) -
    Secao 5 do plano: "jitter gaussiano pequeno por ponto", simula ruido de deteccao do
    MediaPipe. Ao contrario de rotate_z/scale_jitter, somar ruido NAO preserva zero
    exato - remascara por presenca depois, senao um frame ausente ganharia ruido onde
    deveria continuar exatamente zero (quebraria o contrato que normalize_and_mask
    estabelece no gold layer)."""
    landmarks, presence = _split(window)
    noisy = landmarks + rng.normal(scale=sigma, size=landmarks.shape).astype(landmarks.dtype)
    noisy[presence == 0] = 0.0
    return _join(noisy, presence)


def jitter_time(window: np.ndarray, max_shift_steps: float, rng: np.random.Generator) -> np.ndarray:
    """Desloca o eixo temporal por uma fracao aleatoria de passo, ate max_shift_steps
    (Secao 5 do plano: +/-1 frame nativo antes de reamostrar - ver aproximacao explicada
    no docstring do modulo). Reusa a mesma interpolacao do gold layer: linear pros
    landmarks, nearest pra presenca (domain.resampling), remascarando depois pelo mesmo
    motivo de zero_out_absent_hands - um deslocamento pode misturar um frame presente com
    um ausente vizinho."""
    landmarks, presence = _split(window)
    step_ts = np.arange(window.shape[0], dtype=np.float32) / TARGET_FPS
    shift = rng.uniform(-max_shift_steps, max_shift_steps) / TARGET_FPS
    shifted_ts = step_ts + np.float32(shift)

    shifted_landmarks = resample_linear(landmarks, step_ts, shifted_ts)
    shifted_presence = resample_nearest(presence, step_ts, shifted_ts)
    shifted_landmarks[shifted_presence == 0] = 0.0
    return _join(shifted_landmarks, shifted_presence)


def augment_window(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica a augmentacao da Secao 5 do plano sobre UMA janela de treino: cada uma das
    5 transformacoes e aplicada com 50% de chance, independente uma da outra. O plano
    fixa os TIPOS e faixas de cada augmentation, mas nao uma politica de combinacao -
    50% por transformacao e o default escolhido aqui (nao validado empiricamente), pra
    dar uma mistura de janelas mais e menos alteradas em vez de aplicar as 5 sempre
    juntas ou nunca aplicar nenhuma."""
    if rng.random() < 0.5:
        window = mirror_lr(window)
    if rng.random() < 0.5:
        window = jitter_time(window, max_shift_steps=0.5, rng=rng)
    if rng.random() < 0.5:
        window = rotate_z(window, max_degrees=15.0, rng=rng)
    if rng.random() < 0.5:
        window = scale_jitter(window, scale_range=(0.9, 1.1), rng=rng)
    if rng.random() < 0.5:
        window = landmark_noise(window, sigma=0.01, rng=rng)
    return window
