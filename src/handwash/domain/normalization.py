"""Normalizacao por mao (Secao 5 do plano): origem no pulso, escala pela distancia
pulso->MCP-do-dedo-medio. `normalize_and_mask()` e a unidade COMPLETA que sera assada no
grafo exportado no handoff (Dia 2, script 06) - a mesma composicao sera reimplementada la
em ops de TF, pra nao duplicar a logica em Kotlin nem arriscar as duas versoes divergirem
silenciosamente. Usar so normalize_hand() sozinha NAO basta pro grafo exportado: ver o
aviso na propria funcao.

Indices 0 (pulso) e 9 (MCP do dedo medio) sao fixos pela topologia de 21 pontos do
MediaPipe Hand Landmarker - nao mudam entre videos nem entre hand_landmarks/world_landmarks.
"""
import numpy as np

from handwash.config.constants import MIDDLE_MCP_LANDMARK_IDX, WRIST_LANDMARK_IDX

# Achado real ao investigar um CRASH em runtime no modelo TFLite exportado ("DIV failed
# to invoke", tensorflow/lite/kernels/div.cc): 1e-6 evita divisao por zero em float32,
# mas depois de quantizado pra INT8 o proprio piso pode arredondar pra exatamente zero
# (resolucao do INT8 e discreta - um valor real menor que o passo de quantizacao vira 0).
# A distancia pulso->MCP real (maos presentes, dado de treino) nunca fica abaixo de
# ~0.007 e so 0.07% dos casos ficam abaixo de 0.02 - piso de 0.02 fica bem acima da
# resolucao esperada do INT8 pra esse tensor, sem distorcer mao nenhuma de verdade.
_MIN_SCALE = 0.02

# Achado real ao investigar queda de acuracia na quantizacao INT8 (Secao 7.3 do plano):
# quando a mao esta mal rastreada mas nao exatamente ausente (presence=1), a distancia
# pulso->MCP pode ficar pequena sem bater o piso _MIN_SCALE, e a divisao em normalize_hand()
# explode alguns landmarks pra valores extremos (visto ate +-8 no dado real, contra um
# percentil 99.9 de so ~2.1-2.6). Rarissimo (~0.0004% dos valores no split de treino) mas
# o suficiente pra destruir a calibracao MinMax do TFLite INT8 - o range calibrado cresce
# pra acomodar o outlier, e a resolucao efetiva pro dado normal (a maioria) piora bastante.
# Clipar em +-5 (bem acima do p99.9 real, bem abaixo dos outliers observados) resolve sem
# tocar em sinal genuino.
_CLIP_ABS_VALUE = 5.0


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: array[..., 21, 3] (uma ou mais maos/frames, qualquer shape de lote nas
    dimensoes anteriores). Frames zero-preenchidos (mao ausente) permanecem exatamente
    zero: pulso=MCP=0 -> escala cai no piso _MIN_SCALE -> resultado 0/piso=0, sem NaN/Inf.
    Resultado clipado em +-_CLIP_ABS_VALUE - ver nota acima sobre calibracao INT8.
    """
    wrist = landmarks[..., WRIST_LANDMARK_IDX, :]
    middle_mcp = landmarks[..., MIDDLE_MCP_LANDMARK_IDX, :]
    scale = np.maximum(np.linalg.norm(middle_mcp - wrist, axis=-1), _MIN_SCALE)

    centered = landmarks - wrist[..., np.newaxis, :]
    normalized = centered / scale[..., np.newaxis, np.newaxis]
    return np.clip(normalized, -_CLIP_ABS_VALUE, _CLIP_ABS_VALUE)


def zero_out_absent_hands(normalized_landmarks: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """normalized_landmarks: array[n_frames, n_slots, 21, 3]. presence: array[n_frames, n_slots].

    Fecha uma brecha sutil da reamostragem: um frame reamostrado bem na fronteira entre
    deteccao real e um gap longo zero-preenchido interpola uma pose ENCOLHIDA em direcao
    a zero - mas normalize_hand() e invariante a escala uniforme, entao essa pose encolhida
    normaliza de volta pra pose real (o fator de encolhimento cancela nas duas divisoes).
    Sem essa funcao, um frame com presence=0 poderia carregar dados de pose validos
    "vazando" pelo canal errado. Zerar explicitamente por presence garante o contrato:
    presence=0 => landmarks exatamente zero, sempre, sem excecao.

    NAO chamar normalize_hand() sozinha achando que basta - use normalize_and_mask() como
    unidade unica (revisao de codigo confirmou: sem esse passo, presence=0 pode carregar
    uma pose reconstruida da mao real, exatamente nas classes step4/step7 que o plano ja
    espera serem mais confusas por outro motivo - facil de confundir os dois problemas).
    """
    result = normalized_landmarks.copy()
    result[presence == 0] = 0.0
    return result


def normalize_and_mask(landmarks: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Composicao de normalize_hand() + zero_out_absent_hands() - a unidade que deve ser
    reimplementada JUNTA no grafo exportado (Dia 2, script 06). As duas metades separadas
    nao bastam: normalize_hand() sozinha e invariante a escala e pode reconstruir uma pose
    real mesmo num frame com presence=0 (ver zero_out_absent_hands). Usar as duas funcoes
    via chamadas separadas arrisca uma futura reimplementacao (grafo TF ou, por engano,
    Kotlin) esquecer a segunda metade e reabrir esse vazamento silenciosamente - por isso
    esta funcao existe como o unico ponto de entrada esperado fora deste modulo."""
    return zero_out_absent_hands(normalize_hand(landmarks), presence)
