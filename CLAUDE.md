# HigenAI — CLAUDE.md

## Context
HigenAI is the hand-hygiene computer-vision MVP for AI Glasses Brasil 2026.
This file is context for the **data + model track only**. The Android/DAT/runtime architecture is owned by the rest of the team.
The runtime contract is fixed: MediaPipe Hand Landmarker → lightweight temporal classifier → TFLite INT8 → inference on the paired phone; cloud is never part of runtime.
The accepted proposal defines the MVP goal as recognizing the correct **order** of a subset of WHO hand-washing steps, with subject-separated evaluation and audio feedback.
Treat `Plano de Dados e Modelo.md` as the implementation plan and primary source of truth; do not redesign it unless explicitly asked.

## Status
Dia 1 (ambiente, camada bronze, checagem de viabilidade, extração silver completa) e Dia 2 (camada gold, gate de viabilidade TFLite, treino do classificador, export INT8, comparação Keras vs TFLite, pacote de handoff) estão concluídos e validados — ver `DIA1_HANDOFF.md` e `DIA2_HANDOFF.md` antes de tocar em qualquer coisa do Dia 3. Não repita trabalho já feito:
- `data/silver/landmarks/`: 300 vídeos, landmarks extraídos, 0 falhas (Dia 1).
- `data/gold/`: 5.319 janelas normalizadas (treino 4.126 / val 552 / teste 641), 0 vazamento de sujeito (Dia 2).
- `models/checkpoints/handwash_classifier_best.keras`: classificador treinado, 97,19% de acurácia no teste (Dia 2).
- `models/artifacts/`: pacote de handoff completo — `.tflite` quantizado INT8 (96,10% no teste, gate de ~2pp passa), `classes.json`, `MODEL_CARD.md`, `parity_check.py`, `sample_io_pairs.npz` (Dia 2).
O repositório usa clean architecture (`src/handwash/{domain,infrastructure,application,config}`) e arquitetura medalhão (`data/{bronze,silver,gold}`). Scripts `00` a `08` existem e já rodaram com sucesso — ver `DIA2_HANDOFF.md` pro mapeamento exato de cada um (a numeração real diverge um pouco da lista original do plano; `DIA2_HANDOFF.md` explica por quê).

## Scope
Own: dataset inspection, manifest/labels, landmark extraction, temporal windows, augmentation, training, evaluation, calibration, TFLite export, and the Android handoff artifacts.
Do not take ownership of Kotlin/DAT/session/audio/orchestrator design except where a model/data contract must be specified.
Do not replace the planned approach with YOLO, RGB end-to-end video, cloud inference, or a different architecture merely because another approach appears fashionable.
When a choice is explicitly marked “default”, “candidate”, or “open”, validate it experimentally before changing it.
When a decision is marked fixed, preserve it.

## Dataset — confirmed facts
Source: Kaggle `realtimear/hand-wash-dataset`, local copy `D:\Hackton Meta\HandWashDataset`.
The local dataset has 12 folders, 25 videos/folder, 300 MP4 videos, ~1.3 GB.
Subject IDs 001–025 occur consistently across all 12 folders; `G-group` identifies recording batch/physical sink.
Native video rate is 30 fps. G01–G04 are 720×480; G05 is 1920×1080. Clips are ~10–18 s.
Filename regex must tolerate both `G01` and `G_01`: `HandWash_(\d+)_A_(\d+)_G_?(\d+)`.
Use `manifest.csv` as the single source of truth; never re-derive split/class metadata independently in downstream scripts.

## Labels — fixed
Train 7 classes, not 12: merge Left/Right variants while keeping wrist as its own class.
1. `step1_palma_palma`
2. `step2_palma_dorso`
3. `step3_palma_entrelacada`
4. `step4_dorso_dedos`
5. `step5_polegar`
6. `step6_unhas_palma`
7. `step7_pulso`
Do not turn wrist into `Other`; that was only the edi-riga cross-dataset convention.
Class imbalance is exactly 25 vs 50 videos and is handled with class weights, not data deletion.

## Subject split — fixed
Use the auditably fixed split:
`train=[1,3,5,6,8,9,10,11,12,14,15,17,18,20,23,25]`
`val=[4,16,21,22]`
`test=[2,7,13,19,24]`
Hard rule: every window from one subject belongs to exactly one split. Assert this in code.
Keep G-group representation visible when evaluating; do not let overlapping windows create subject leakage.

## Landmark pipeline — fixed contract
Use MediaPipe Tasks Hand Landmarker, `running_mode=VIDEO`, `num_hands=2`, with real timestamps.
Per-video NPZ:
`hand_landmarks [frames,2,21,3]`, `world_landmarks [frames,2,21,3]`, `presence [frames,2]`, `timestamps_sec [frames]`.
Slots are left/right according to MediaPipe handedness, independent of which hand is active.
Gaps ≤5 frames: linear interpolation. Long/end gaps: zero-fill and `presence=0`. Log missing-detection rate per video/class.
Resample by timestamp to 15 fps; do not blindly take every second frame.
Training window: 2.0 s = 30 steps at 15 fps. Train stride 0.5 s; val/test stride 1.0 s.
Never shuffle temporal frame order.

## Features and normalization — confirmed, with load-bearing refinements
Input is 128 features/frame: 126 landmark coordinates + 2 presence flags; window shape `(30,128)`.
Normalize each hand around wrist landmark 0 and scale by `||landmark[9]-landmark[0]||`, floored at `max(scale, 0.02)`, result clipped to `±5.0`. Both refinements were found necessary on Day 2, not cosmetic — do not simplify them away:
- A smaller floor (e.g. `1e-6`, perfectly safe in float32) can round to exactly zero once the graph is quantized to INT8, crashing the TFLite interpreter at runtime (`DIV failed to invoke`) on frames with a barely-tracked hand. `0.02` is well above the INT8 resolution for this tensor and below the real minimum observed wrist→MCP distance (~0.007), so it never distorts a genuine hand.
- An unclipped result lets rare near-degenerate hand poses (~0.0004% of frames) produce extreme normalized values (observed up to ±8, vs a real p99.9 of ~2.1–2.6), which blows up the INT8 calibration range and silently degrades accuracy for every class, not just the outlier frame.
Normalization is embedded in the exported graph (`infrastructure/normalization_layer.py`, TF-ops reimplementation confirmed to match `domain/normalization.py` with zero numerical difference on real data), not duplicated in Kotlin. `domain/normalization.py::normalize_and_mask()` is the single entry point to call — never call `normalize_hand()` alone, see the module docstring for why.
Primary training input: `world_landmarks` — confirmed on Day 2: 97.19% test accuracy (Keras FP32) / 96.10% (TFLite INT8), both above the stretch reference. Image-coordinate ablation still not done; treat the choice as empirically successful, not comparatively validated against the alternative.

## Model — confirmed
Primary (confirmed working end-to-end on Day 2): Conv1D(64,k=5) → LayerNorm → Conv1D(64,k=5) → Dropout(.3) → GRU(64, `unroll=True`) → Dropout(.4) → Dense(32) → Dropout(.3) → Dense(7,softmax), 88,967 parameters.
`unroll=True` on the GRU is required, not optional — confirmed empirically: the dynamic GRU (`unroll=False`) builds and trains fine but fails builtin-only TFLite conversion with `tf.TensorListReserve ... requires element_shape to be static`. This was the only change needed; Option A converted, trained, and quantized successfully on the first architecture tried.
Fallback B: GRU(48,seq) → Dropout(.3) → GRU(32) → Dropout(.4) → Dense(7). Never used — Option A did not overfit (train/val/test accuracy stayed close).
Fallback C: Conv1D×3 → GlobalAveragePooling1D → Dense(32) → Dense(7), for builtin-only TFLite compatibility. Never needed — see `unroll=True` above.
Do not introduce a larger model without explicit instruction; the dataset has only 25 subjects and overfitting is a primary risk.

## Regularization and training — confirmed
Dropout 0.3–0.4; L2=1e-4 on GRU/Dense; label smoothing=0.1; early stopping on val loss, patience 8–10 — confirmed: training stops naturally around epoch 30–38 out of the 80-epoch ceiling, never needs the full budget.
API note confirmed empirically before writing training code, still true as of this Keras version (3.15): `SparseCategoricalCrossentropy` does **not** accept `label_smoothing` here. Use `CategoricalCrossentropy(label_smoothing=0.1)` with one-hot-encoded targets instead (`tf.keras.utils.to_categorical`). Re-check this if the TF/Keras version ever changes.
Augment normalized landmarks with `mirror_lr`, `jitter_time`, `rotate_z` ±10–15°, `scale_jitter` [0.9,1.1], and small landmark noise — implemented in `domain/augmentation.py`, applied on-the-fly per batch (fresh random draw every epoch, train split only, never a fixed augmented set on disk). `jitter_time` is a post-resample micro time-shift on the already-built window (reusing `domain.resampling`'s interpolation), not a literal re-resample from native silver frames — the gold layer doesn't persist the pre-window per-video sequence, and reprocessing from silver every epoch would be far more expensive for a "small" augmentation. Each of the 5 augmentations is applied independently with 50% probability per window per epoch — an unvalidated default; the plan fixes the augmentation types and ranges, not a combination policy.
`mirror_lr` must flip x and swap left/right slots. Never alter temporal order.
Adam lr=1e-3; ReduceLROnPlateau factor=.5, patience=4; batch=32; 60–80 epoch ceiling.
Compute class weights from the **bronze** manifest (`data/bronze/manifest.csv`, video-level counts, exactly 25 vs 50) — not the gold/window-level manifest. "manifest.csv" is ambiguous between the two; this interpretation was chosen because the Labels section of this file describes the imbalance in video counts. Do not copy edi-riga's positional `get_weights_dict()`.

## Evaluation gates — confirmed results (Day 2)
Report test accuracy overall **and per class**, confusion matrix, and per-video accuracy by majority vote across windows — all in `models/checkpoints/training_report.json` (Keras) and `tflite_vs_keras_report.json` (comparison).
Calibration matters because confidence drives the orchestrator: mean softmax confidence is 0.918 on correct predictions vs 0.691 on incorrect ones (Keras FP32, test set) — clearly separated. Not re-measured separately for the quantized model. Temperature scaling was not needed.
Hard MVP floor: test accuracy > 0.7511 (Lulla et al., 2021) — **passed**: 97.19% (Keras) / 96.10% (TFLite INT8).
Stretch reference: 93.5% reported by GhostHandEgoNet on Kaggle — **exceeded** by both versions.
Compare Keras FP32 vs exported TFLite INT8 on the same test windows. Investigate any >~2 percentage-point accuracy drop, especially class-specific collapse — **this happened for real, not hypothetically**: the first export attempt dropped 4+ percentage points overall with up to 30pp collapse concentrated in `step1_palma_palma`, traced to two real bugs in how `normalize_hand()` behaves under INT8 quantization (see Features/normalization section). Fixed; final drop is 1.09pp, gate passes. **If the normalization formula changes again for any reason, re-run `scripts/07_evaluate_tflite_vs_keras.py` before trusting a new export — a passing gate is not permanent once the formula changes.**
Expected difficult classes are step 4 and step 7 — **partially confirmed, partially corrected**: step7_pulso did show real confusion with step4_dorso_dedos as predicted, but step4 itself was the single easiest class (100%). The classes that actually struggled most were step1_palma_palma and step3_palma_entrelacada (both "palm-to-palm" variants, mutually confused in FP32, and the pair most sensitive to INT8 quantization specifically) — not anticipated by the original plan. This is a documented experiment result correcting the original hypothesis, not a contradiction to explain away.

## TFLite and handoff — confirmed (Day 2)
Before full training, test the primary layer types with a dummy model using builtin-only TFLite conversion — done in `scripts/smoke_test_tflite.py` (not `05_...`, see Status section note on numbering), passed with `GRU(unroll=True)` on the first architecture tried.
Avoid `SELECT_TF_OPS`/Flex unless all planned fallbacks fail — never needed.
Export full-integer INT8 weights + activations while keeping float32 input/output — confirmed pattern: do not set `inference_input_type`/`inference_output_type` on the converter; leaving them unset keeps float32 I/O by default alongside `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]`.
Representative data: **stratified by class, ~43 windows/class (301 total) from TRAIN only**, refined from the original "~300 generic" plan. Plain proportional sampling from the full train pool under-represents the two smaller classes (step1/step3, 16/192 train videos each) and calibrates them badly. Counterintuitively, *larger* representative sets (150+/class) made accuracy measurably worse, not better, even after the normalization bugs above were fixed — root cause not fully isolated (suspected: some internal Conv1D/GRU activation range, not the input landmarks themselves). **Do not casually increase the representative dataset size without re-running the Keras-vs-TFLite comparison to check** — bigger is not automatically safer here.
Handoff delivered in `models/artifacts/`: `handwash_step_classifier.tflite`, frozen `classes.json`, `MODEL_CARD.md`, `parity_check.py` (tested — passes), `sample_io_pairs.npz` (3 windows, distinct classes, generated from the same exported model).
The model card documents exact shapes/dtypes, 2 s/15 fps time-based windowing, embedded normalization (confirmed exact parity between the numpy and TF-ops implementations, zero numerical difference), buffer contract, calibration numbers, per-class accuracy, and the known external-camera → egocentric domain gap — see `models/artifacts/MODEL_CARD.md` for full detail, `DIA2_HANDOFF.md` for the execution narrative behind the numbers.

## Workflow and evidence
Start with dataset probes and manifest validation; measure actual extraction speed on 10–20 videos before processing all 300.
Run the builtin-only TFLite smoke test before spending time on the full training run.
Keep experiments reproducible: fixed splits, explicit seeds/configs, saved metrics, confusion matrices, and model artifacts.
Use the edi-riga repository as a reference for dataset handling and prior baselines, not as code to copy blindly.
When reporting results, distinguish **confirmed facts**, **experiment results**, and **hypotheses**. Never present an unvalidated default as a proven fact.
If a task conflicts with this plan, stop and surface the conflict instead of silently changing the design.
**This TF/Keras version (tensorflow 2.21 / keras 3.15) has non-obvious API and numeric behavior — verify empirically with a small throwaway probe before writing production training/export code, don't assume.** Day 2 hit three real surprises this way, each caught *before* it became a shipped bug: `GRU(unroll=False)` silently fails TFLite builtin-only conversion; `SparseCategoricalCrossentropy` doesn't accept `label_smoothing` in this Keras version; and a `1e-6` epsilon that's perfectly safe in float32 can round to exactly zero under INT8 quantization and crash the interpreter at runtime. Numeric safety margins tuned for float32 do not automatically carry over once quantization is involved — re-check them, don't assume.
