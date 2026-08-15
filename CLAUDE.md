# HigenAI — CLAUDE.md

## Context
HigenAI is the hand-hygiene computer-vision MVP for AI Glasses Brasil 2026.
This file is context for the **data + model track only**. The Android/DAT/runtime architecture is owned by the rest of the team.
The runtime contract is fixed: MediaPipe Hand Landmarker → lightweight temporal classifier → TFLite INT8 → inference on the paired phone; cloud is never part of runtime.
The accepted proposal defines the MVP goal as recognizing the correct **order** of a subset of WHO hand-washing steps, with subject-separated evaluation and audio feedback.
Treat `Plano de Dados e Modelo.md` as the implementation plan and primary source of truth; do not redesign it unless explicitly asked.

## Status
Dia 1 (ambiente, camada bronze, checagem de viabilidade, extração silver completa) está concluído e validado — ver `DIA1_HANDOFF.md` antes de tocar em qualquer coisa do Dia 2. Não repita trabalho já feito: os 300 vídeos já têm landmarks extraídos em `data/silver/landmarks/` (0 falhas). O repositório usa clean architecture (`src/handwash/{domain,infrastructure,application,config}`) e arquitetura medalhão (`data/{bronze,silver,gold}`) — `gold/` ainda está vazio, é o próximo passo do Dia 2.

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

## Features and normalization — fixed/default
Input is 128 features/frame: 126 landmark coordinates + 2 presence flags; window shape `(30,128)`.
Normalize each hand around wrist landmark 0 and scale by `||landmark[9]-landmark[0]||`; normalization must be embedded in the exported graph, not duplicated in Kotlin.
Primary training input: `world_landmarks`. Treat this as a hypothesis to validate, with image-coordinate ablation only if time permits.

## Model — planned
Primary: Conv1D(64,k=5) → LayerNorm → Conv1D(64,k=5) → Dropout(.3) → GRU(64) → Dropout(.4) → Dense(32) → Dropout(.3) → Dense(7,softmax), ~89K parameters.
Fallback B: GRU(48,seq) → Dropout(.3) → GRU(32) → Dropout(.4) → Dense(7).
Fallback C: Conv1D×3 → GlobalAveragePooling1D → Dense(32) → Dense(7), for builtin-only TFLite compatibility.
Do not introduce a larger model without explicit instruction; the dataset has only 25 subjects and overfitting is a primary risk.

## Regularization and training — fixed plan
Dropout 0.3–0.4; L2=1e-4 on GRU/Dense; label smoothing=0.1; early stopping on val loss, patience 8–10.
Augment normalized landmarks with `mirror_lr`, `jitter_time`, `rotate_z` ±10–15°, `scale_jitter` [0.9,1.1], and small landmark noise.
`mirror_lr` must flip x and swap left/right slots. Never alter temporal order.
Adam lr=1e-3; ReduceLROnPlateau factor=.5, patience=4; batch=32; 60–80 epoch ceiling.
Compute class weights from `manifest.csv`; do not copy edi-riga's positional `get_weights_dict()`.

## Evaluation gates
Report test accuracy overall **and per class**, confusion matrix, and per-video accuracy by majority vote across windows.
Calibration matters because confidence drives the orchestrator: compare confidence on correct vs incorrect predictions; temperature scaling is allowed only on validation data if needed.
Hard MVP floor: test accuracy > 0.7511 (Lulla et al., 2021).
Stretch reference: 93.5% reported by GhostHandEgoNet on Kaggle; this is a research reference, not a requirement to alter the plan.
Compare Keras FP32 vs exported TFLite INT8 on the same test windows. Investigate any >~2 percentage-point accuracy drop, especially class-specific collapse.
Expected difficult classes are step 4 and step 7; document confusion patterns rather than assuming they are bugs.

## TFLite and handoff
Before full training, test the primary layer types with a dummy model using builtin-only TFLite conversion.
Avoid `SELECT_TF_OPS`/Flex unless all planned fallbacks fail.
Export full-integer INT8 weights + activations while keeping float32 input/output.
Representative data: ~300 windows from TRAIN only; never validation/test.
Handoff must include: `handwash_step_classifier.tflite`, frozen `classes.json`, `MODEL_CARD.md`, an independent parity check, and 2–3 real test IO pairs.
The model card must document exact shapes/dtypes, 2 s/15 fps time-based windowing, embedded normalization, buffer contract, calibration, per-class accuracy, and the known external-camera → egocentric domain gap.

## Workflow and evidence
Start with dataset probes and manifest validation; measure actual extraction speed on 10–20 videos before processing all 300.
Run the builtin-only TFLite smoke test before spending time on the full training run.
Keep experiments reproducible: fixed splits, explicit seeds/configs, saved metrics, confusion matrices, and model artifacts.
Use the edi-riga repository as a reference for dataset handling and prior baselines, not as code to copy blindly.
When reporting results, distinguish **confirmed facts**, **experiment results**, and **hypotheses**. Never present an unvalidated default as a proven fact.
If a task conflicts with this plan, stop and surface the conflict instead of silently changing the design.
