"""Constantes de dominio - janela temporal, fps alvo, esquema de classes, splits por sujeito.

Todos os numeros aqui vem do plano de dados/modelo aprovado (Secoes 1-3). Nao sao "chutes":
- NATIVE_FPS_EXPECTED: confirmado por parsing direto de boxes MP4 em amostra cobrindo todos os G-groups.
- SUBJECT_SPLITS / SUBJECT_TO_GGROUP / SINK_BY_SUBJECT: derivados de HandWashDataset/HandwashList.txt,
  verificados duas vezes de forma independente (agente de exploracao + leitura direta do arquivo).
"""

# --- Video / reamostragem ---
NATIVE_FPS_EXPECTED = 30.0
TARGET_FPS = 15.0  # fps do stream MEDIUM do DAT em runtime - treino precisa casar com isso
WINDOW_SEC = 2.0
WINDOW_STEPS = int(WINDOW_SEC * TARGET_FPS)  # 30
TRAIN_STRIDE_SEC = 0.5
VAL_TEST_STRIDE_SEC = 1.0

# --- Normalizacao por mao e contrato de features (Secao 5 do plano) ---
# Indices fixos pela topologia de 21 pontos do MediaPipe Hand Landmarker.
WRIST_LANDMARK_IDX = 0
MIDDLE_MCP_LANDMARK_IDX = 9
NUM_LANDMARK_COORDS = 2 * 21 * 3  # 2 maos x 21 pontos x (x,y,z) = 126
WINDOW_FEATURE_DIM = NUM_LANDMARK_COORDS + 2  # + 2 presence flags = 128, contrato fixo

# --- Esquema de classes: 7 classes finais, Left/Right mesclado, pulso como classe propria ---
RAW_FOLDER_TO_CLASS = {
    "Step_1": "step1_palma_palma",
    "Step_2_Left": "step2_palma_dorso",
    "Step_2_Right": "step2_palma_dorso",
    "Step_3": "step3_palma_entrelacada",
    "Step_4_Left": "step4_dorso_dedos",
    "Step_4_Right": "step4_dorso_dedos",
    "Step_5_Left": "step5_polegar",
    "Step_5_Right": "step5_polegar",
    "Step_6_Left": "step6_unhas_palma",
    "Step_6_Right": "step6_unhas_palma",
    "Step_7_Left": "step7_pulso",
    "Step_7_Right": "step7_pulso",
}

# Ordem CONGELADA - indice do softmax = posicao nesta lista. NUNCA reordenar depois
# que isso for entregue ao time Android (contrato de handoff, Secao 8 do plano).
CLASSES = [
    "step1_palma_palma",
    "step2_palma_dorso",
    "step3_palma_entrelacada",
    "step4_dorso_dedos",
    "step5_polegar",
    "step6_unhas_palma",
    "step7_pulso",
]

# --- Split por sujeito, ciente de G-group (Secao 3 do plano) ---
SUBJECT_SPLITS = {
    "train": [1, 3, 5, 6, 8, 9, 10, 11, 12, 14, 15, 17, 18, 20, 23, 25],
    "val": [4, 16, 21, 22],
    "test": [2, 7, 13, 19, 24],
}

# --- G-group -> pia/setup de gravacao (fonte: HandwashList.txt) ---
SUBJECT_TO_GGROUP = {}
for _id in range(1, 9):
    SUBJECT_TO_GGROUP[_id] = "G01"
for _id in range(9, 12):
    SUBJECT_TO_GGROUP[_id] = "G02"
for _id in range(12, 15):
    SUBJECT_TO_GGROUP[_id] = "G03"
for _id in range(15, 18):
    SUBJECT_TO_GGROUP[_id] = "G04"
for _id in range(18, 26):
    SUBJECT_TO_GGROUP[_id] = "G05"

SINK_BY_SUBJECT = {}
for _id in range(1, 6):
    SINK_BY_SUBJECT[_id] = "DarkSteelSink1"
for _id in range(6, 9):
    SINK_BY_SUBJECT[_id] = "LightSteelSink1"
for _id in range(9, 12):
    SINK_BY_SUBJECT[_id] = "CeramicSink1"
for _id in range(12, 15):
    SINK_BY_SUBJECT[_id] = "SmallCeramicSink6"
for _id in range(15, 18):
    SINK_BY_SUBJECT[_id] = "LightGraniteSink1"
for _id in range(18, 26):
    SINK_BY_SUBJECT[_id] = "CeramicSink2"

del _id

# --- Checagem de viabilidade (gate, Secao 1.3 do plano) ---
# Criterio de gate = at_least_one_hand_rate (pelo menos uma mao visivel por frame).
# both_hands_rate e reportado como informativo, nao como gate - varios passos da
# tecnica OMS sobrepoem/entrelacam as maos por definicao, entao uma taxa baixa
# de "as duas maos ao mesmo tempo" e esperada e nao invalida o dado.
VIABILITY_MIN_AT_LEAST_ONE_HAND_RATE = 0.80
