# handwash_model — HigenAI · trilha de dados + modelo

Reconhecimento da técnica OMS de higienização das mãos a partir de landmarks de mão (MediaPipe) + classificador temporal leve (Conv1D + GRU), para o projeto HigenAI (Programa AI Glasses Brasil 2026 — CEIA/UFG · FUNAPE · Meta).

Este repositório cobre a **trilha de dados + modelo** apenas. A trilha de Android/DAT/orquestrador/voz vive em repositório separado do time.

---

## Status de progresso

| Etapa | Status | Resultado |
|---|---|---|
| Ambiente (conda `higenai`, Python 3.11) | ✅ Concluído | mediapipe 1.0.1 · tensorflow 2.21.0 · opencv 5.0.0 |
| Camada Bronze (manifest 300 vídeos) | ✅ Concluído | 300 vídeos indexados, 7 classes, split por sujeito auditado, 0 vazamentos |
| Checagem de viabilidade (gate MediaPipe) | ✅ Concluído | 90,6% `at_least_one_hand_rate` geral · 11/12 pastas acima de 80% |
| Camada Silver (extração de landmarks) | ✅ Concluído | 300/300 .npz · 0 falhas · 0 NaN/Inf · fps 30.04–30.23 |
| Camada Gold (janelas normalizadas) | ✅ Concluído | 5.319 janelas (treino 4.126 / val 552 / teste 641) · 0 vazamento de sujeito |
| Smoke test TFLite builtin-only | ✅ Concluído | GRU dinâmico falha a conversão; `GRU(unroll=True)` converte e roda |
| Treino do classificador | ✅ Concluído | Conv1D+GRU, 88.967 params · **97,19%** teste · **100%** por vídeo (voto majoritário) |
| Exportação TFLite INT8 | ✅ Concluído | **96,10%** teste (queda de 1,09pp, dentro do gate de ~2pp) · normalização embutida no grafo |
| Pacote de handoff Android | ✅ Concluído | `.tflite` + `classes.json` + `MODEL_CARD.md` + `parity_check.py` (testado) + `sample_io_pairs.npz` |
| Testes unitários (`domain/`) | ✅ Concluído | 52/52 passando |
| **Teste de domain gap (câmera egocêntrica)** | 🔲 Dia 3 | Gravar vídeo próprio, rodar pelo pipeline, medir queda de acurácia |

---

## Arquitetura

### Dados — arquitetura medalhão

```
data/
  bronze/    manifest.csv          → 300 vídeos indexados (sujeito, classe, split, G-group)
  silver/    manifest.csv          → métricas de extração por vídeo
             landmarks/*.npz       → landmarks frame-a-frame (gitignored — derivado, ~grande)
  gold/      manifest.csv          → 1 linha por janela (vídeo, split, classe, split_row_idx)
             classes.json          → as 7 classes, ordem congelada
             {train,val,test}/windows.npz  → arrays X (N,30,128) + y (N,), gitignored (derivado)
```

### Código — clean architecture

```
src/handwash/
  domain/          → lógica pura, sem I/O: parsing de nome, esquema de classes, split por
                     sujeito, gaps, amostragem, reamostragem 15fps, janelamento, normalização
                     por mão, augmentação (5 tipos), pesos de classe
  infrastructure/  → adapters de I/O: OpenCV, MediaPipe Hand Landmarker, fábrica de modelos
                     Keras, conversor TFLite, camada de normalização em ops de TF (export)
  application/     → casos de uso (build_manifest, extract_landmarks, build_windows,
                     check_tflite_viability, train_classifier, export_tflite,
                     evaluate_tflite_vs_keras, build_handoff_package)
  config/          → constantes (CLASSES, SUBJECT_SPLITS, janela temporal, contrato de
                     features) e caminhos
scripts/           → wrappers finos de CLI: 00–04 (Dia 1-2), smoke_test_tflite.py (gate,
                     sem número), 05–08 (treino, export, comparação, handoff — Dia 2)
tests/             → 52 testes unitários da camada domain
```

### Contrato de classes (7 classes, fixo — índice = posição no softmax)

| Índice | Classe | Pastas de origem |
|---|---|---|
| 0 | `step1_palma_palma` | Step_1 |
| 1 | `step2_palma_dorso` | Step_2_Left + Step_2_Right |
| 2 | `step3_palma_entrelacada` | Step_3 |
| 3 | `step4_dorso_dedos` | Step_4_Left + Step_4_Right |
| 4 | `step5_polegar` | Step_5_Left + Step_5_Right |
| 5 | `step6_unhas_palma` | Step_6_Left + Step_6_Right |
| 6 | `step7_pulso` | Step_7_Left + Step_7_Right |

### Split por sujeito (fixo, auditável)

```python
train = [1,3,5,6,8,9,10,11,12,14,15,17,18,20,23,25]   # 16 sujeitos → 192 vídeos
val   = [4,16,21,22]                                    #  4 sujeitos →  48 vídeos
test  = [2,7,13,19,24]                                  #  5 sujeitos →  60 vídeos
```

Cada janela derivada de um sujeito pertence a exatamente um split — verificado por assert em código.

---

## Setup

```bash
D:\Miniconda\Scripts\conda.exe create -n higenai python=3.11 -y
D:\Miniconda\Scripts\activate.bat higenai
pip install -r requirements.txt
pip install -e .
```

Baixar o bundle do MediaPipe em `models/mediapipe/hand_landmarker.task` (não versionado):
[https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

O dataset bruto (`D:\Hackton Meta\HandWashDataset`, 300 .mp4, ~1,3 GB) fica **fora** do repo — não é duplicado.

---

## Pipeline completo

### Dia 1 — já executado

```bash
python scripts/00_probe_dataset.py      # confirma fps (30fps) e resolução (720×480 / 1920×1080)
python scripts/01_build_manifest.py     # camada bronze: manifest.csv com 300 vídeos
python scripts/02_viability_check.py    # gate: MediaPipe detecta bem nesse dataset?
python scripts/03_extract_landmarks.py  # camada silver: 300 .npz de landmarks (~37 min)
pytest tests/ -q                        # 24 testes domain passando
```

### Dia 2 — já executado

```bash
python scripts/04_build_windows.py           # gold: 5.319 janelas, norm. por mão embutida
python scripts/smoke_test_tflite.py          # gate: GRU(unroll=True) converte builtin-only
python scripts/05_train_classifier.py        # treino: 97,19% teste, 100% por vídeo
python scripts/06_export_tflite.py           # INT8 full-integer, normalização embutida no grafo
python scripts/07_evaluate_tflite_vs_keras.py  # Keras vs TFLite: 96,10% teste (queda 1,09pp)
python scripts/08_build_handoff_package.py   # monta models/artifacts/ (classes.json + sample IO)
pytest tests/ -q                             # 52 testes domain passando
```

Reconstruir do zero (04 → 08) leva ~1-2 minutos no total — os scripts são idempotentes.

### Dia 3 — próximo passo

Testar o domain gap câmera de pia (dataset de treino) → câmera egocêntrica (óculos, nunca testada):
gravar um vídeo próprio, rodar pelo mesmo pipeline de extração/gold, avaliar contra o modelo treinado.

---

## Modelo (confirmado, Dia 2)

**Arquitetura primária — 88.967 parâmetros, a única testada, converteu e treinou de primeira:**
```
Input (30 passos, 128 features)
  └─ Conv1D(64, k=5) → LayerNorm → Conv1D(64, k=5) → Dropout(0.3)
  └─ GRU(64, unroll=True) → Dropout(0.4)
  └─ Dense(32) → Dropout(0.3) → Dense(7, softmax)
```
`unroll=True` é obrigatório: a GRU dinâmica (`unroll=False`) falha a conversão TFLite builtin-only
(`tf.TensorListReserve ... requires element_shape to be static`) — confirmado empiricamente.

**Entrada por frame:** 2 mãos × 21 landmarks × 3 coords + 2 flags de presença = **128 features**
**Janela temporal:** 2,0s → 30 passos a 15fps (reamostrado por timestamp, não por índice)
**Normalização:** por mão, origem no pulso (landmark 0), escala pela dist. pulso→MCP (landmark 9),
piso de escala `0.02` e resultado limitado a `±5.0` — embutida no grafo exportado, com paridade
numérica exata confirmada entre a versão Python e a versão TF do grafo. Os dois valores acima
(piso e limite) não são cosméticos: sem eles, a quantização INT8 podia travar em runtime ou perder
acurácia — ver [`models/artifacts/MODEL_CARD.md`](models/artifacts/MODEL_CARD.md) para os números.

**Fallbacks documentados, nunca usados:**
- Opção B: GRU(48, seq=True) → GRU(32) → Dense(7) — só se A tivesse overfitado (não overfitou)
- Opção C: Conv1D×3 → GlobalAvgPool → Dense(32) → Dense(7) — só se GRU não convertesse (converteu)

**Resultado real (teste, sujeitos nunca vistos em treino):**

| | Keras FP32 | TFLite INT8 |
|---|---|---|
| Acurácia geral (por janela) | 97,19% | 96,10% |
| Acurácia por vídeo (voto majoritário, 60 vídeos) | 100% | não medido separadamente |

Piso obrigatório (> 75,1%, Lulla et al. 2021) e referência esticada (93,5%, GhostHandEgoNet,
IJPRAI 2026) **superados nas duas versões**. Classes mais fracas: `step1_palma_palma` e
`step3_palma_entrelacada` (confundidas entre si) — não `step4`/`step7` como o plano original
apostava; ver o model card para a matriz de confusão completa.

---

## Handoff para o time Android (`models/artifacts/`)

Entregue ao fim do Dia 2:

| Arquivo | Descrição |
|---|---|
| `handwash_step_classifier.tflite` | modelo quantizado INT8 (float32 I/O), 522KB |
| `classes.json` | lista ordenada das 7 classes (índice = softmax) — congelada |
| `MODEL_CARD.md` | shapes/dtypes, windowing 2s/15fps, norm. embutida, acurácia por classe, limitações |
| `parity_check.py` | script p/ o time Android validar o .tflite independentemente — testado, passa |
| `sample_io_pairs.npz` | 3 janelas reais de teste (classes diferentes) com saída esperada |

Falta: o time Android rodar `parity_check.py` do lado deles e confirmar. Domain gap
(câmera de pia → óculos egocêntricos) ainda não testado — ver Status acima.

---

## Decisões de arquitetura fixas

Ver [`CLAUDE.md`](CLAUDE.md) para o registro completo das decisões que não podem ser alteradas sem revisão explícita (esquema de classes, split, contrato de entrada/saída do modelo, configuração TFLite).
