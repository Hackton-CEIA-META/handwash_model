# handwash_model — HigenAI · trilha de dados + modelo

Reconhecimento da técnica OMS de higienização das mãos a partir de landmarks de mão (MediaPipe) + classificador temporal leve (Conv1D + GRU), para o projeto HigenAI (Programa AI Glasses Brasil 2026 — CEIA/UFG · FUNAPE · Meta).

Este repositório cobre a **trilha de dados + modelo** apenas. A trilha de Android/DAT/orquestrador/voz vive em repositório separado do time.

---

## Resumo pra quem só vai abrir uma vez

- **Modelo treinado e testado** (Conv1D+GRU, 88.967 parâmetros) sobre 300 vídeos do dataset Kaggle `realtimear/hand-wash-dataset`: **97,19%** de acurácia por janela, **100%** por vídeo com voto majoritário — acima do piso obrigatório da literatura (75,11%, Lulla et al. 2021) e da referência esticada (93,5%, GhostHandEgoNet).
- **Exportado e quantizado pra INT8** (o que de fato roda no dispositivo embarcado): **96,10%** de acurácia, queda de só 1,09pp, artefato final de **510KB**.
- **Testado contra vídeo próprio da equipe** (câmera de celular, o mais perto possível do ponto de vista dos óculos): a acurácia cai pra **~49%** — a maior limitação conhecida do protótipo hoje, medida e documentada com honestidade, não escondida. Detalhe completo em [`notebooks/02_domain_gap_limitations.ipynb`](notebooks/02_domain_gap_limitations.ipynb).
- Uma tentativa de mitigar isso (misturar vídeo próprio no treino) foi feita, medida, e **não funcionou** — decisão consciente de reverter e manter o modelo em produção treinado só com dado Kaggle, arquitetura limpa. Ver seção "Dia 3" abaixo.
- Os dois notebooks executados, com gráficos e vídeo real do pipeline rodando (esqueleto de mão + predição ao vivo), estão em [`notebooks/`](notebooks/) — é o material mais rápido pra avaliar o projeto visualmente.

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
| **Teste de domain gap (câmera egocêntrica)** | ✅ Concluído | **~49%** de acurácia (queda de ~48pp vs. Kaggle) — limitação real, documentada |
| **Tentativa de fine-tuning com dado próprio** | ✅ Testado, revertido | Não melhorou a generalização — modelo em produção continua só-Kaggle |
| Notebooks de análise (viabilidade + limitações) | ✅ Concluído | [`notebooks/01_model_viability.ipynb`](notebooks/01_model_viability.ipynb) · [`notebooks/02_domain_gap_limitations.ipynb`](notebooks/02_domain_gap_limitations.ipynb) |
| Coordenação com time Android (`parity_check.py`) | 🔲 Em aberto | Pacote pronto em `models/artifacts/`, falta validação do lado deles |

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
  custom/    manifest.csv          → vídeo próprio da equipe (Dia 3, teste de domain gap)
             landmarks/*.npz       → gitignored, mesma lógica do silver
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
                     evaluate_tflite_vs_keras, build_handoff_package, evaluate_domain_gap)
  config/          → constantes (CLASSES, SUBJECT_SPLITS, janela temporal, contrato de
                     features) e caminhos
scripts/           → wrappers finos de CLI: 00–04 (Dia 1-2), smoke_test_tflite.py (gate,
                     sem número), 05–08 (treino, export, comparação, handoff — Dia 2),
                     09 (teste de domain gap — Dia 3)
notebooks/         → análise exploratória e resultados (viabilidade + limitações), com
                     gráficos e vídeo real do pipeline rodando — Dia 3
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

## O que foi feito em cada dia

### Dia 1 — dados, viabilidade, extração

Ambiente configurado, camada bronze construída (300 vídeos indexados, split por sujeito auditado contra vazamento), gate de viabilidade confirmando que o MediaPipe detecta mão de forma confiável nesse dataset (90,6% de `at_least_one_hand_rate`), e extração completa de landmarks pros 300 vídeos (0 falhas).

```bash
python scripts/00_probe_dataset.py      # confirma fps (30fps) e resolução (720×480 / 1920×1080)
python scripts/01_build_manifest.py     # camada bronze: manifest.csv com 300 vídeos
python scripts/02_viability_check.py    # gate: MediaPipe detecta bem nesse dataset?
python scripts/03_extract_landmarks.py  # camada silver: 300 .npz de landmarks (~37 min)
pytest tests/ -q                        # 24 testes domain passando
```

### Dia 2 — treino, quantização, handoff

Camada gold (janelas normalizadas, 5.319 no total), gate de conversão TFLite validado antes de investir tempo em treino real, classificador treinado e avaliado (97,19% teste, 100% por vídeo), exportado com quantização INT8 full-integer (96,10% teste, queda de 1,09pp — dois bugs reais de quantização encontrados e corrigidos no caminho, documentados no `MODEL_CARD.md`), e pacote de handoff completo pro time Android.

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

### Dia 3 — o teste que faltava: domain gap, e uma tentativa honesta de mitigação

Todo o treino e teste do Dia 2 usa câmera fixa de pia (dataset Kaggle) — a visão egocêntrica dos óculos nunca tinha sido testada. Esse era o maior risco não coberto do protótipo, e era previsto desde a proposta original como "o passo seguinte necessário".

**O que foi feito:**
1. Gravação de 21 vídeos próprios com celular (3 tomadas por passo OMS, fundos/ângulos variados, câmera segurada perto do rosto/peito — a aproximação mais próxima possível do ponto de vista dos óculos).
2. Avaliação contra o modelo já treinado (`scripts/09_evaluate_domain_gap.py`, sem retreinar nada): **49,5%** de acurácia por janela (Keras) / **49,1%** (TFLite), queda de quase 48 pontos percentuais — a detecção de mão pelo MediaPipe continua confiável (98,3% de `at_least_one_hand_rate`), então o problema é inteiramente de generalização do classificador, não de visão.
3. Investigação de causa: o padrão de erro é consistente entre tomadas — `step2`, `step5` e `step6` seguram razoavelmente bem fora do domínio; `step1`, `step4` e `step7` colapsam quase por completo. Hipótese mais provável: a normalização por punho+escala corrige posição e tamanho da mão, mas não o ângulo de onde a câmera está olhando.
4. **Tentativa de mitigação**: renomear os vídeos próprios pro padrão do dataset e mesclá-los ao treino (retreino do zero + fine-tuning a partir do modelo original), medindo contra um segundo integrante da equipe nunca visto em nenhum treino. **Nenhuma das duas variantes melhorou a acurácia nesse sujeito de teste** — ambas ficaram levemente piores que o modelo original. Interpretação mais provável: uma pessoa a mais, com poucas tomadas, não é volume/diversidade suficiente pra ensinar invariância de ângulo.
5. **Decisão**: reverter a mescla por completo (dados e código) e manter o modelo de produção treinado só com o dataset Kaggle — mesmo checkpoint do Dia 2, intocado. A tentativa fica documentada como aprendizado, não como resultado escondido.

```bash
python scripts/09_evaluate_domain_gap.py   # avalia vídeo próprio contra o modelo já treinado
```

**Resultado completo, com gráficos, matriz de confusão, tabela vídeo a vídeo e vídeo real do pipeline rodando** (mostrando visualmente onde o esqueleto de mão continua certo mas a classe erra): [`notebooks/02_domain_gap_limitations.ipynb`](notebooks/02_domain_gap_limitations.ipynb). A contraparte de viabilidade, com a mesma profundidade de evidência sobre o resultado no domínio Kaggle: [`notebooks/01_model_viability.ipynb`](notebooks/01_model_viability.ipynb).

---

## Modelo (confirmado, Dia 2 — em produção)

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

**Resultado no domínio de treino (teste Kaggle, sujeitos nunca vistos em treino):**

| | Keras FP32 | TFLite INT8 |
|---|---|---|
| Acurácia geral (por janela) | 97,19% | 96,10% |
| Acurácia por vídeo (voto majoritário, 60 vídeos) | 100% | não medido separadamente |

Piso obrigatório (> 75,1%, Lulla et al. 2021) e referência esticada (93,5%, GhostHandEgoNet,
IJPRAI 2026) **superados nas duas versões**. Classes mais fracas: `step1_palma_palma` e
`step3_palma_entrelacada` (confundidas entre si) — não `step4`/`step7` como o plano original
apostava; ver o model card para a matriz de confusão completa.

**Resultado fora do domínio de treino (vídeo próprio, câmera de celular — ver Dia 3 acima):**

| | Keras FP32 | TFLite INT8 |
|---|---|---|
| Acurácia geral (por janela) | 49,5% | 49,1% |
| Acurácia por vídeo (voto majoritário, 21 vídeos) | 52,4% | 52,4% |

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

Falta: o time Android rodar `parity_check.py` do lado deles e confirmar. O modelo entregue é o
treinado só com Kaggle (checkpoint do Dia 2) — o domain gap pra câmera egocêntrica já foi medido
(Dia 3, acima) e continua como limitação conhecida, não resolvida nesta rodada.
