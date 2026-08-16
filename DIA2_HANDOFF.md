# HigenAI — Handoff Dia 2 → Dia 3

**Data:** 2026-08-16. **Status:** Dia 2 100% concluído e validado.

## Como usar este documento

Ordem de leitura recomendada numa sessão nova: `CLAUDE.md` (regras fixas, já atualizado com os fatos do Dia 2) → `DIA1_HANDOFF.md` (o que aconteceu no Dia 1) → **este arquivo** (o que aconteceu no Dia 2, resultado real, onde exatamente o Dia 3 começa) → `Plano de Dados e Modelo.md` (detalhe técnico de cada decisão original, consultar por seção) → `models/artifacts/MODEL_CARD.md` (contrato exato do modelo entregue, é o que o time Android vai ler).

Este documento é o relato de execução do Dia 2, no mesmo espírito do `DIA1_HANDOFF.md`: o plano dizia o que fazer, este arquivo conta o que de fato aconteceu ao rodar — incluindo dois bugs reais encontrados e corrigidos que não estavam previstos em nenhum documento anterior.

---

## 0. Onde tudo está

- **Repositório**: `D:\Hackton Meta\handwash_model`. Commit-base do Dia 1: `a3e16c5`. Todo o trabalho do Dia 2 está no working tree, **nenhum commit foi feito** — combinado desde o Dia 1, continua valendo.
- **Ambiente**: mesmo do Dia 1, sem mudança — conda env `higenai`, `D:\Miniconda\envs\higenai\python.exe`, `tensorflow==2.21.0`, `keras==3.15.1` (a versão exata do Keras importa — ver Seção 3, item 4).

---

## 1. O que foi executado, passo a passo, com resultado real

### 1.1 Camada Gold (`scripts/04_build_windows.py`)

**O que é:** transforma os landmarks silver (300 `.npz`, 30fps nativo) em janelas de treino prontas — reamostragem por timestamp a 15fps, normalização por mão, split por sujeito aplicado.

**Como funciona:** `domain/resampling.py` (interpolação linear pros landmarks, nearest pra presença — funções puras reaproveitadas depois em 3 lugares diferentes: gold layer, augmentação `jitter_time`, e dataset representativo do export), `domain/windowing.py` (fatiamento em janelas de 30 passos, descarta sobra em vez de preencher com padding), `domain/normalization.py` (`normalize_and_mask()`, ver Seção 1.4 sobre por que isso virou uma função fundida), `domain/splits.py` (`assert_no_cross_split_subjects()`, reforçado — ver Seção 1.2). `application/build_windows.py` orquestra tudo.

**Resultado real:** 5.319 janelas — treino 4.126, val 552, teste 641. Todos os 300 vídeos contribuíram pelo menos 1 janela. Zero vazamento de sujeito entre splits (verificado de forma independente, não só pelo assert interno). Roda em **~1,7 segundos** pros 300 vídeos.

**Achado sobre o stride de treino**: 0,5s não divide exato em passos de 15fps (0,5×15=7,5) — arredonda pra 8 passos (~0,533s real). Matematicamente inevitável dado os dois números fixos (15fps, 0,5s), documentado no código, não é bug.

### 1.2 Revisão de código antes de treinar

Antes de gastar tempo com treino real, rodei uma revisão formal (8 ângulos de análise em paralelo, depois 14 candidatos verificados individualmente, todos por sub-agentes independentes) sobre o diff completo do Dia 2 até aquele ponto. Achados reais, corrigidos:

- **`assert_no_cross_split_subjects` só checava consistência interna, não contra o `SUBJECT_SPLITS` real** — um manifest desatualizado passaria batido. Corrigido pra comparar contra `split_for_subject()` diretamente.
- **`normalize_hand()` + `zero_out_absent_hands()` viraram `normalize_and_mask()`** — as duas funções precisam viajar juntas (uma normaliza, a invariante a escala; a outra fecha um vazamento sutil onde presence=0 podia carregar uma pose reconstruída). Separá-las arriscava uma reimplementação futura (o grafo TF do export, por exemplo) esquecer a segunda metade.
- **`build_windows()` simplificado** — removidos dois contadores manuais (`per_split_y`, `split_cursors`) que sincronizavam por convenção, não por construção. Um risco real se o loop fosse paralelizado no futuro.
- **Script do gate TFLite renomeado**: a revisão achou um `README.md` que eu não tinha lido, com uma convenção de nomes (`scripts/smoke_test_tflite.py`, sem número) diferente da que eu tinha inventado (`05_tflite_builtin_ops_check.py`). Renomeado — isso libera "05" pro treino, como o README já documentava.
- Deixados de lado conscientemente (baixa severidade): duplicação de `_read_bronze_manifest()` com o arquivo já validado do Dia 1 (não vale o risco de tocar em código do Dia 1 por 3 linhas), um dict repetido 4x no gate TFLite (estilo, não bug), um wraparound de índice negativo inatingível com os dados reais.

**39 → 51 testes** depois dessa rodada (domínio: resampling, windowing, normalização, split leakage).

### 1.3 Gate de viabilidade TFLite (`scripts/smoke_test_tflite.py`)

**O que é:** confirma que a arquitetura principal (Conv1D+GRU+Dense) converte pra TFLite builtin-only ANTES de treinar de verdade — testado com pesos aleatórios, não treinados, porque o teste é sobre compatibilidade de operação, não acurácia.

**Resultado real**: Opção A como está (GRU dinâmico) **falha** a conversão com `tf.TensorListReserve ... requires element_shape to be static` — exatamente o suspeito que o plano previu. Opção A com `GRU(unroll=True)` **converte e roda**, 88.967 parâmetros (bate com a estimativa de ~89K do plano), ~456KB em float32 não quantizado. Não precisou da Opção C.

**Infraestrutura criada**: `infrastructure/model_factory.py` (`build_primary_model(gru_unroll=...)`, `build_fallback_c_model()`), `infrastructure/tflite_converter.py`, `application/check_tflite_viability.py` (a cadeia de fallback exata da Seção 7.1 do plano: A → A+unroll → C, nunca Flex).

### 1.4 Treino do classificador (`scripts/05_train_classifier.py`)

**O que é:** treina a Opção A vencedora (GRU unroll=True) com os dados gold, class weights, augmentação on-the-fly, avaliação completa.

**Duas surpresas de API confirmadas empiricamente antes de escrever o código de produção** (mesmo probe descartável usado pro gate TFLite):
1. `SparseCategoricalCrossentropy` **não aceita** `label_smoothing` nesta versão do Keras (3.15) — só `CategoricalCrossentropy` aceita, e exige rótulo one-hot. Troquei a perda e converti os rótulos.
2. `tf.keras.utils.Sequence` e `tf.keras.utils.PyDataset` são **a mesma classe** nesta versão — usei `PyDataset` diretamente sem problema.

**Infraestrutura criada**: `domain/augmentation.py` (as 5 augmentations do plano — `mirror_lr`, `jitter_time`, `rotate_z`, `scale_jitter`, `landmark_noise` — todas preservando o contrato "presence=0 ⇒ zero exato"; `jitter_time` reusa `domain.resampling` em vez de reprocessar do silver, ver `CLAUDE.md`), `domain/class_weights.py` (calculado do manifest **bronze**, nível de vídeo — decisão registrada no `CLAUDE.md` porque "manifest.csv" era ambíguo), `application/train_classifier.py`.

**Resultado real**: 97,19% de acurácia no teste (por janela), **100% por vídeo com voto majoritário** (60/60), 37 épocas (parou sozinho antes do teto de 80). Supera o piso obrigatório (75,11%) e a referência esticada (93,5%). Calibração: confiança média 0,918 nos acertos vs. 0,691 nos erros — bem separada.

**Por classe**, todas ≥89%: as mais fracas foram `step1_palma_palma` (89,4%, confundida com `step3`) e `step3_palma_entrelacada` (92,5%) — ambas variantes de "palma com palma", visualmente próximas. `step7_pulso` (96,0%) mostrou a confusão com `step4` que o CLAUDE.md previa, mas `step4_dorso_dedos` em si saiu perfeito (100%) — a hipótese original (step4+step7 mais difíceis) só se confirmou parcialmente.

### 1.5 Export TFLite INT8 — a parte que quase saiu quebrada (`scripts/06_export_tflite.py`)

**O que é:** compõe [normalização embutida] + [modelo treinado] num grafo único, quantiza INT8 full-integer mantendo entrada/saída float32.

**Decisão de arquitetura que teve que ser resolvida**: o CLAUDE.md fixa que a normalização tem que estar **embutida no grafo exportado**, não duplicada em Kotlin — mas o modelo foi treinado sobre dado JÁ normalizado (gold layer). Resolvido criando `infrastructure/normalization_layer.py` (`NormalizeAndMaskLayer`, reimplementação em ops de TF de `domain/normalization.py`) e compondo `[raw_input] → NormalizeAndMaskLayer → [modelo treinado, pesos reais reusados] → output`. **Confirmado por paridade exata (diff numérico zero)** entre a camada TF e a versão numpy antes de exportar. Isso também exigiu expor `build_video_features(video_stem, normalize=False)` em `build_windows.py` (antes privada, `_build_video_features`) pro export conseguir montar o dataset representativo com landmarks BRUTOS — o mesmo formato que o grafo vai receber em produção.

**A investigação real** (Seção 7.3 do plano manda "investigar queda >~2pp", não só reportar):

1. Primeira comparação: queda agregada de 1,87pp (dentro do gate) — mas escondendo um colapso de **7,6pp só em `step1`**.
2. Tentei corrigir achando que era cobertura de vídeo insuficiente no dataset representativo → **piorou** (4,37pp, step1 caiu 30,3pp).
3. Tentei estratificar por classe → **piorou mais ainda** (4,52pp, step1 31,8pp).
4. Diagnóstico decisivo: comparei o modelo composto em **float32, sem quantizar nenhuma** contra o modelo original — bateram **exatamente** (100% de concordância, diferença zero). Isso provou que o bug não estava na minha composição do grafo, era 100% efeito da quantização INT8.
5. Investigando os valores normalizados usados na calibração: o range MinMax crescia de [-5,4, 2,7] pra [-8,1, 7,0] conforme o dataset representativo aumentava, mas o percentil 99,9% ficava estável em ~2,1-2,6 — outliers raros (mão quase degenerada, distância pulso→MCP minúscula) estavam destruindo a calibração.
6. Durante um teste de tamanhos maiores de dataset representativo, o modelo **travou em runtime**: `DIV failed to invoke`. O piso `_MIN_SCALE=1e-6` (seguro em float32) podia arredondar pra exatamente zero sob quantização INT8.
7. Duas correções reais, não ajuste de parâmetro: `_MIN_SCALE` subiu pra `0.02` (acima da resolução esperada do INT8, abaixo de qualquer distância real observada) e `normalize_hand()` ganhou `clip(±5.0)`. Reconstruí gold, retreinei, reexportei, recomparei a cada mudança.

**Resultado final**: queda caiu de 4+pp (com crash) pra **1,09pp, dentro do gate de ~2pp, sem crash**. Dataset representativo final: 43 janelas/classe (301 total), estratificado — a escolha que passou na validação real, não um número arbitrário do plano.

**Achado documentado mas não resolvido**: mesmo com os dois bugs corrigidos, datasets representativos MAIORES (150+/classe) ainda pioram a acurácia de forma monotônica — o oposto do que calibração saudável faria. Suspeita: alguma ativação interna (Conv1D/GRU), não mais a entrada de landmarks. Está no `MODEL_CARD.md` e no `CLAUDE.md` como aviso explícito pra não aumentar o dataset representativo sem re-testar.

### 1.6 Comparação Keras vs TFLite (`scripts/07_evaluate_tflite_vs_keras.py`)

Roda os dois modelos sobre as MESMAS 641 janelas de teste (Keras recebe dado já normalizado de `data/gold/test/`; TFLite recebe as mesmas janelas reconstruídas em forma bruta, com um assert que confirma que normalizar o bruto reproduz exatamente o gold — garante que a comparação é justa). Resultado final: Keras 97,19%, TFLite 96,10%, queda 1,09pp, concordância direta entre os dois modelos 98,28%.

### 1.7 Pacote de handoff (`scripts/08_build_handoff_package.py`)

`models/artifacts/`: `handwash_step_classifier.tflite` (522KB), `classes.json` (congelado, copiado do gold), `MODEL_CARD.md` (documenta os dois bugs de quantização encontrados, contrato exato de entrada/saída, resultados por classe, limitações), `parity_check.py` (testado — roda e passa, 3/3 amostras batem exatamente), `sample_io_pairs.npz` (3 janelas reais de teste, classes diferentes, com a saída esperada do `.tflite`).

### 1.8 Testes

**52 testes**, todos passando (`pytest tests/ -q`). Cobrem domínio inteiro: parsing, classes, splits (incluindo o assert reforçado), gaps, amostragem, resampling, windowing, normalização (incluindo o teste de regressão do vazamento presence/normalização E o teste do clip de outlier), augmentação (todas as 5, cada uma verificando que o contrato presence=0⇒zero se mantém), pesos de classe.

---

## 2. Achados que o Dia 3 precisa saber (não redescobrir)

1. **step1/step3, não step4/step7, são o ponto fraco real.** O CLAUDE.md original apostava em step4/step7 como as classes mais difíceis (fisicamente mais difíceis de executar). step7 confirmou parcialmente (confundida com step4), mas step4 em si saiu perfeita. Quem realmente struggles é o par step1/step3 (ambas "palma com palma") — tanto em float32 quanto (bem mais) sob quantização INT8. Se o Dia 3 for mexer no modelo, é aqui que vale prestar atenção.
2. **Os dois bugs de quantização (`_MIN_SCALE=0.02`, `clip(±5.0)`) são estruturais, não cosméticos.** Reverter qualquer um dos dois reabre um crash real em runtime (não só uma queda de acurácia) — documentado com o mecanismo exato no `CLAUDE.md` e `MODEL_CARD.md`. Se `normalize_hand()` for tocado de novo por qualquer motivo, rodar `scripts/07_evaluate_tflite_vs_keras.py` de novo antes de confiar no export.
3. **Dataset representativo maior != melhor**, mesmo com os bugs corrigidos — fenômeno real, documentado, causa raiz não totalmente isolada. Não aumentar "43/classe" sem re-testar.
4. **Domain gap câmera externa → egocêntrica continua 100% não testado.** Todo o treino/teste é sobre o dataset Kaggle (câmera fixa de pia). Isso é o maior risco não coberto que resta — ver checklist do Dia 3 abaixo.
5. **Numeração de scripts final**: `00`-`04` (Dia 1-2), `smoke_test_tflite.py` (sem número, gate), `05` (treino), `06` (export), `07` (comparação), `08` (handoff). Diverge um pouco da lista original do `Plano de Dados e Modelo.md` — o `README.md` do repo é a fonte mais atual pra isso, releia se for criar um script novo.
6. **A versão exata do TF/Keras (2.21/3.15) tem comportamento não óbvio** — 3 surpresas reais encontradas só testando empiricamente antes de escrever código de produção (GRU unroll, label_smoothing, o crash de quantização). Ver a nota nova no final do `CLAUDE.md` (seção Workflow).

---

## 3. O que NÃO foi feito ainda — exatamente onde o Dia 3 começa

- **Teste de domain gap com vídeo próprio** (Plano, seção Verificação, Dias 3-4): gravar um vídeo de lavagem de mãos com câmera de celular, rodar pelo mesmo pipeline de extração, comparar contra o desempenho no dataset Kaggle. Isso é o item mais importante que falta — tudo até aqui foi validado só contra câmera fixa de pia.
- **Vídeo outlier nunca teve inspeção visual humana**: `HandWash_025_A_03_G_05` (sujeito 25, 1,94% de detecção) — mencionado no `DIA1_HANDOFF.md`, ainda não verificado.
- **Ablação `world_landmarks` vs. coordenadas de imagem normalizadas** (Plano, Seção 9, item 3): não testada — a escolha por `world_landmarks` teve sucesso empírico (97%+) mas nunca foi comparada contra a alternativa.
- **Acurácia por vídeo (voto majoritário) só foi medida no Keras FP32** (100%), não separadamente na versão TFLite — a acurácia por janela já bate o gate, então isso é inferência razoável, não medição direta.
- **K-fold agrupado por sujeito** (Plano, Seção 5, item 7 — stretch, não obrigatório): não feito, só é relevante se sobrar tempo.
- Coordenação com o time Android pra rodar `parity_check.py` do lado deles e validar o `.tflite` de forma independente — o pacote está pronto, falta o time confirmar.

## 4. Checklist do Dia 3 (mapeado pro roadmap original)

1. **Domain gap**: gravar 1+ vídeo de teste com câmera de celular (ou mais perto possível do setup real dos óculos), rodar `scripts/03_extract_landmarks.py` → `04_build_windows.py` (sem normalizar de novo — o pipeline já faz isso) → avaliar contra o modelo treinado. Documentar honestamente se a acurácia cai, e quanto.
2. Se sobrar tempo: inspeção visual do vídeo outlier (item 3 acima), ablação `world_landmarks` vs. imagem normalizada.
3. Suporte ao time Android na integração do pacote de handoff (`models/artifacts/`) — responder dúvidas de contrato, não assumir a implementação Kotlin.

## 5. Avisos importantes

- **Nenhum commit foi feito** — igual ao Dia 1, continua valendo pro Dia 3 também, a menos que combinem o contrário.
- `CLAUDE.md` já foi atualizado com todos os fatos confirmados do Dia 2 (normalização, arquitetura, treino, gates, TFLite) — é a fonte mais rápida de consultar antes deste documento.
- Reconstruir qualquer coisa (`04` → `05` → `06` → `07`) é rápido (~1-2 min no total) se precisar — os scripts são idempotentes, sobrescrevem sem pedir confirmação.
