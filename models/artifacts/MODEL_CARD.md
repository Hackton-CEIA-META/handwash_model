# MODEL_CARD — handwash_step_classifier.tflite

HigenAI (AI Glasses Brasil 2026). Classificador temporal leve que reconhece qual dos 7 passos
da técnica OMS de higienização das mãos está sendo executado, a partir de landmarks de mão
(MediaPipe Hand Landmarker) já extraídos — este modelo não vê pixels, só pose.

## Contrato de entrada/saída (exato)

| | Shape | Dtype | Semântica |
|---|---|---|---|
| Entrada | `(1, 30, 128)` | `float32` | 30 passos de tempo × 128 features/passo |
| Saída | `(1, 7)` | `float32` | Softmax — probabilidade por classe, já normalizada (soma ≈ 1.0) |

**As 128 features por passo são landmarks BRUTOS, não normalizados** — 126 coordenadas
(2 mãos × 21 pontos MediaPipe × [x,y,z] `world_landmarks`, slot 0 = mão esquerda / slot 1 =
mão direita pela lateralidade do MediaPipe, layout fixo) + 2 flags de presença (1.0 =
detectado, 0.0 = ausente/preenchido). **A normalização está embutida no grafo — ver seção
abaixo. Não normalizar do lado Android antes de chamar o modelo.**

Índice do softmax = posição em `classes.json` (congelado, não reordenar):
```
0 step1_palma_palma      4 step5_polegar
1 step2_palma_dorso      5 step6_unhas_palma
2 step3_palma_entrelacada 6 step7_pulso
3 step4_dorso_dedos
```

## Janela temporal — contrato de buffer

- Janela = **2.0 segundos, reamostrados a 15fps por TIMESTAMP** (30 passos), não por
  contagem ingênua de frame — o fps real de captura (DAT via Bluetooth) não é um metrônomo
  perfeito, então o buffer precisa interpolar por tempo, igual ao pipeline de treino fez.
- **Não rodar inferência até o buffer de 2.0s estar completamente cheio** (30 passos). O
  modelo nunca viu janela parcial/padded em treino — alimentá-lo com uma janela incompleta
  produz saída sem sentido, não um erro visível.
- A ordem temporal dos 30 passos importa (sinal real, não features independentes) —
  nunca embaralhar.

## Normalização — embutida no grafo, não reimplementar em Kotlin

O grafo exportado tem, como primeira camada, a normalização por mão (origem no pulso,
escala pela distância pulso→MCP do dedo médio), a mesma fórmula usada em treino
(`domain/normalization.py::normalize_and_mask`, reimplementada em ops de TF em
`infrastructure/normalization_layer.py`). **Confirmado por paridade exata (diff numérico
zero) entre as duas implementações** antes de exportar — ver `parity_check.py`.

Isso significa: o app Android manda landmarks crus do MediaPipe (resampleados/janelados,
mas sem nenhuma matemática de normalização) direto pro `Interpreter`. Toda a normalização
acontece dentro do `.tflite`.

## Quantização INT8 — dois bugs reais encontrados e corrigidos

Full-integer (pesos + ativações INT8), entrada/saída mantidas em `float32` (contrato mais
simples pro Android — chamada padrão de `Interpreter`, sem matemática de escala/zero-point
do lado Kotlin). Dataset representativo: 43 janelas por classe (301 total, só do split de
TREINO, nunca validação/teste).

Duas coisas que pareciam "detalhe de implementação" em float32 viraram bugs reais sob
quantização — documentado aqui porque um ajuste ingênuo nesses dois valores reabre os
problemas silenciosamente:

1. **Piso de escala (`_MIN_SCALE`) tinha que sobreviver à quantização, não só a float32.**
   `1e-6` evita divisão por zero em `float32` (nunca é literalmente zero), mas depois de
   quantizado pra INT8 (resolução discreta) esse piso podia arredondar pra exatamente
   zero — causando **crash em runtime** (`DIV failed to invoke`) em frames com mão mal
   rastreada. Corrigido para `0.02` (a distância pulso→MCP real nunca fica abaixo de
   ~0.007 nos dados de treino; só 0.07% dos casos ficam abaixo de 0.02 — piso seguro sob
   INT8 sem distorcer mão nenhuma de verdade).
2. **Saída de `normalize_hand()` não tinha teto.** Mãos mal rastreadas (presence=1, mas
   quase sem distância entre pulso e MCP) produziam landmarks normalizados extremos
   (observado até ±8, contra um percentil 99.9% real de só ~2.1–2.6). Raríssimo
   (~0.0004% dos valores) mas o suficiente pra destruir a calibração MinMax da
   quantização — o range calibrado esticava pra acomodar o outlier, piorando a resolução
   efetiva pra todo o resto. Corrigido com `clip(±5.0)` dentro de `normalize_hand()`.

Com os dois corrigidos: acurácia caiu de **97,19% (Keras FP32) para 96,10% (TFLite
INT8) no teste — 1,09 pontos percentuais**, dentro do gate de ~2pp do plano. Antes da
correção, a queda chegava a 4,4pp com colapso de até 30pp numa classe específica.

**Achado ainda em aberto, documentado mas não resolvido**: datasets representativos
MAIORES (150+, 300+ por classe) pioram a acurácia de forma monotônica, o oposto do
esperado numa calibração saudável — mesmo já com os dois bugs acima corrigidos. Suspeita:
sensibilidade de quantização em alguma ativação interna (Conv1D/GRU) ainda não isolada,
não mais relacionada à entrada/escala. A escolha de 43/classe é a que passou a validação
real, não um valor arbitrário — não aumentar o dataset representativo sem re-testar.

## Resultados (teste, sujeitos nunca vistos em treino)

| Métrica | Keras FP32 | TFLite INT8 |
|---|---|---|
| Acurácia geral (por janela) | 97,19% | 96,10% |
| Acurácia por vídeo (voto majoritário, 60 vídeos) | 100% | — (não medido, ver nota) |

**Por classe:**

| Classe | Keras | TFLite | Queda (pp) |
|---|---|---|---|
| step1_palma_palma | 89,4% | 83,3% | 6,1 |
| step2_palma_dorso | 100% | 100% | 0,0 |
| step3_palma_entrelacada | 92,5% | 88,7% | 3,8 |
| step4_dorso_dedos | 100% | 100% | 0,0 |
| step5_polegar | 100% | 100% | 0,0 |
| step6_unhas_palma | 97,2% | 97,2% | 0,0 |
| step7_pulso | 96,0% | 94,9% | 1,0 |

step1/step3 (variantes de "palma com palma") concentram toda a queda — são as duas
classes mais mutuamente confundidas já em float32 (matriz de confusão em
`training_report.json`), plausivelmente uma fronteira de decisão mais estreita entre elas.
**O CLAUDE.md antecipava step4/step7 como as classes mais difíceis** (fisicamente mais
difíceis de executar); o resultado real aponta step1/step3 como o ponto fraco — resultado
de experimento, não a hipótese original confirmada ao pé da letra. Documentar a diferença,
não forçar a narrativa pra bater com a expectativa.

**Comparação com a literatura**: piso obrigatório do MVP é 75,11% (Lulla et al., 2021) —
superado com folga grande. Referência esticada é 93,5% (GhostHandEgoNet) — também superada
(97,19% em float32, 96,10% já quantizado).

## Calibração (confiança do softmax)

Confiança média (Keras FP32) nas previsões corretas: **0,918**. Nas incorretas: **0,691**.
Separação clara — confiança alta é um sinal razoavelmente confiável de acerto, útil pro
orquestrador usar como threshold. Números medidos em float32; a versão quantizada não foi
recalibrada separadamente (`training_report.json` tem a matriz de confusão completa).

## Limitações conhecidas

- **Domain gap câmera externa → egocêntrica, ainda não testado.** Todo o treino/teste usa
  vídeos gravados com câmera fixa de pia (dataset Kaggle `realtimear/hand-wash-dataset`),
  não a câmera egocêntrica dos óculos. Esse gap é reconhecido desde a proposta original —
  planejado pra ser testado nos Dias 3-4 com um vídeo gravado pelo próprio time.
- **Vídeo outlier conhecido no treino**: `HandWash_025_A_03_G_05` (sujeito 25,
  step2_palma_dorso) tem 1,94% de taxa de detecção — caso genuinamente difícil (câmera
  zenital, mãos ensaboadas), não arquivo corrompido. Nunca teve inspeção visual humana.
- Acurácia por vídeo (voto majoritário) só foi medida no modelo Keras FP32, não
  re-verificada na versão TFLite — a acurácia por janela já bate o gate, e o voto
  majoritário deve suavizar ainda mais qualquer ruído introduzido pela quantização, mas
  isso é inferência, não medição direta.
- Dataset pequeno (25 sujeitos). Regularização (dropout, L2, label smoothing, augmentação,
  early stopping) aplicada especificamente por causa disso — ver `training_report.json`
  para a config completa.

## Arquitetura (Opção A, vencedora do gate de viabilidade TFLite)

```
Input (30, 128) - landmarks BRUTOS + presence
  -> NormalizeAndMaskLayer (embutida, sem peso treinável)
  -> Conv1D(64, k=5) -> LayerNorm -> Conv1D(64, k=5) -> Dropout(0.3)
  -> GRU(64, unroll=True) -> Dropout(0.4)
  -> Dense(32) -> Dropout(0.3) -> Dense(7, softmax)
```
88.967 parâmetros. `GRU(unroll=True)` é obrigatório para a conversão TFLite builtin-only —
a versão dinâmica (`unroll=False`) falha com `tf.TensorListReserve ... requires
element_shape to be static` (confirmado empiricamente, ver `scripts/smoke_test_tflite.py`).
Conversão builtin-only, sem `SELECT_TF_OPS`/Flex.

## Arquivos deste pacote

- `handwash_step_classifier.tflite` — o modelo, pronto pra uso
- `classes.json` — as 7 classes, ordem congelada (índice = posição no softmax)
- `sample_io_pairs.npz` — 3 janelas reais de teste (classes diferentes) + saída esperada
  do `.tflite`, pra escrever um teste unitário sem precisar rodar Python/TF
- `parity_check.py` — valida o `.tflite` de forma independente contra `sample_io_pairs.npz`
