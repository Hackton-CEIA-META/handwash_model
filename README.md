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
| Testes unitários (`domain/`) | ✅ Concluído | 24/24 passando |
| **Camada Gold (janelas normalizadas)** | 🔲 Dia 2 | Próximo passo: `scripts/04_build_windows.py` |
| Smoke test TFLite builtin-only | 🔲 Dia 2 | Antes do treino real (Conv1D+GRU dummy → conversão) |
| Treino do classificador | 🔲 Dia 2 | Conv1D+GRU ~89K params · val/test por sujeito |
| Exportação TFLite INT8 | 🔲 Dia 2 | float32 I/O · builtin-ops-only · comparação Keras vs TFLite |
| Pacote de handoff Android | 🔲 Dia 2 | `.tflite` + `classes.json` + `MODEL_CARD.md` + IO de exemplo |

---

## Arquitetura

### Dados — arquitetura medalhão

```
data/
  bronze/    manifest.csv          → 300 vídeos indexados (sujeito, classe, split, G-group)
  silver/    manifest.csv          → métricas de extração por vídeo
             landmarks/*.npz       → landmarks frame-a-frame (gitignored — derivado, ~grande)
  gold/      (vazio — Dia 2)       → janelas (30 passos × 128 features), prontas p/ treino
```

### Código — clean architecture

```
src/handwash/
  domain/          → lógica pura, sem I/O: parsing de nome, esquema de classes,
                     split por sujeito, preenchimento de gaps, amostragem
  infrastructure/  → adapters de I/O: OpenCV, MediaPipe Hand Landmarker
  application/     → casos de uso (build_manifest, extract_landmarks, build_windows...)
  config/          → constantes (CLASSES, SUBJECT_SPLITS, janela temporal) e caminhos
scripts/           → wrappers finos de CLI (00–03 executados; 04–07 são Dia 2)
tests/             → 24 testes unitários da camada domain
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

### Dia 2 — próximos passos (em ordem)

```bash
python scripts/04_build_windows.py      # camada gold: janelas 2s/30 passos, norm. por mão
python scripts/smoke_test_tflite.py     # PRIMEIRO: Conv1D+GRU dummy → conversão builtin-only
python scripts/05_train_classifier.py   # treino: Conv1D+GRU ~89K params, class weights do manifest
python scripts/06_export_tflite.py      # INT8 full-integer, float32 I/O, repr. dataset só do train
python scripts/07_evaluate_tflite.py    # tabela Keras vs TFLite por classe + handoff artifacts
```

---

## Modelo planejado

**Arquitetura primária (~89K params):**
```
Input (30 passos, 128 features)
  └─ Conv1D(64, k=5) → LayerNorm → Conv1D(64, k=5) → Dropout(0.3)
  └─ GRU(64) → Dropout(0.4)
  └─ Dense(32) → Dropout(0.3) → Dense(7, softmax)
```

**Entrada por frame:** 2 mãos × 21 landmarks × 3 coords + 2 flags de presença = **128 features**  
**Janela temporal:** 2,0s → 30 passos a 15fps (reamostrado por timestamp, não por índice)  
**Normalização:** por mão, origem no pulso (landmark 0), escala pela dist. pulso→MCP (landmark 9) — embutida no grafo exportado

**Fallbacks documentados:**
- Opção B: GRU(48, seq=True) → GRU(32) → Dense(7) — se A overfitar
- Opção C: Conv1D×3 → GlobalAvgPool → Dense(32) → Dense(7) — se GRU não converter em builtin-only TFLite

**Alvo de acurácia:** piso obrigatório > 75,1% (Lulla et al. 2021) · referência 93,5% (GhostHandEgoNet, IJPRAI 2026)

---

## Handoff para o time Android (`models/artifacts/`)

Arquivos que serão entregues ao fim do Dia 2:

| Arquivo | Descrição |
|---|---|
| `handwash_step_classifier.tflite` | modelo quantizado INT8 (float32 I/O) |
| `classes.json` | lista ordenada das 7 classes (índice = softmax) — congelada |
| `MODEL_CARD.md` | shapes/dtypes, windowing 2s/15fps, norm. embutida, acurácia por classe, limitações |
| `parity_check.py` | script p/ o time Android validar o .tflite independentemente |
| `sample_io_pairs.npz` | 2–3 janelas reais de teste com saída esperada |

---

## Decisões de arquitetura fixas

Ver [`CLAUDE.md`](CLAUDE.md) para o registro completo das decisões que não podem ser alteradas sem revisão explícita (esquema de classes, split, contrato de entrada/saída do modelo, configuração TFLite).
