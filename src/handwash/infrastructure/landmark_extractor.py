"""Adapter de I/O: extracao de landmarks de mao via MediaPipe Tasks (VIDEO mode).

Sem logica de negocio aqui - so fala com o video e com o MediaPipe.
Preenchimento de gaps, normalizacao, etc. ficam na camada domain.

IMPORTANTE: o HandLandmarker em modo VIDEO exige uma sequencia de timestamps
estritamente crescente durante toda a vida da INSTANCIA - nao "por video"
processado. Por isso extract_video_landmarks() cria e fecha sua propria
instancia a cada chamada, em vez de reaproveitar uma instancia entre varios
videos - reaproveitar causa "ValueError: Input timestamp must be monotonically
increasing" assim que o segundo video comeca do timestamp 0 de novo.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

NUM_HAND_SLOTS = 2
NUM_LANDMARKS = 21
NUM_COORDS = 3


@dataclass(frozen=True)
class VideoLandmarkExtraction:
    hand_landmarks: np.ndarray  # float32[n_frames, 2, 21, 3] - imagem-normalizado
    world_landmarks: np.ndarray  # float32[n_frames, 2, 21, 3] - 3D metrico
    presence: np.ndarray  # uint8[n_frames, 2]
    timestamps_sec: np.ndarray  # float32[n_frames]


def _slot_for_handedness(handedness_category_name: str) -> int:
    # slot 0 = mao esquerda, slot 1 = mao direita - layout fixo independente
    # de qual mao esta mais ativa no clipe (Secao 4.2 do plano).
    return 0 if handedness_category_name == "Left" else 1


def extract_video_landmarks(
    video_path: Path,
    model_path: Path,
    num_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
) -> VideoLandmarkExtraction:
    """Extrai landmarks de mao de UM video, frame a frame, em modo VIDEO do MediaPipe.

    Cria e fecha sua propria instancia de HandLandmarker (ver nota de modulo) -
    nao chamar de forma a compartilhar uma instancia entre videos diferentes.

    Ordem de criacao/fechamento deliberada (achado em revisao de codigo): o
    landmarker e criado ANTES de abrir o video, dentro do try/finally que o
    fecha - assim, se cv2.VideoCapture falhar depois, o landmarker ja criado
    nao vaza. E vice-versa: se o proprio create_from_options falhar, nada mais
    foi aberto ainda.
    """
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
    )

    landmarker = HandLandmarker.create_from_options(options)
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Nao foi possivel abrir o video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            hand_landmarks_seq = []
            world_landmarks_seq = []
            presence_seq = []
            timestamps_seq = []

            frame_idx = 0
            last_timestamp_ms = -1
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                timestamp_ms = int((frame_idx / fps) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    # salvaguarda defensiva: garante estritamente crescente mesmo se
                    # o truncamento de int() colidir em fps muito alto.
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                frame_hand = np.zeros((NUM_HAND_SLOTS, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
                frame_world = np.zeros((NUM_HAND_SLOTS, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
                frame_presence = np.zeros((NUM_HAND_SLOTS,), dtype=np.uint8)

                if result.hand_landmarks:
                    for hand_idx, handedness in enumerate(result.handedness):
                        slot = _slot_for_handedness(handedness[0].category_name)
                        lm = result.hand_landmarks[hand_idx]
                        wlm = result.hand_world_landmarks[hand_idx]
                        frame_hand[slot] = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                        frame_world[slot] = np.array([[p.x, p.y, p.z] for p in wlm], dtype=np.float32)
                        frame_presence[slot] = 1

                hand_landmarks_seq.append(frame_hand)
                world_landmarks_seq.append(frame_world)
                presence_seq.append(frame_presence)
                timestamps_seq.append(frame_idx / fps)

                frame_idx += 1
        finally:
            cap.release()
    finally:
        landmarker.close()

    if not hand_landmarks_seq:
        raise ValueError(f"Nenhum frame legivel em {video_path} - arquivo vazio ou corrompido?")

    return VideoLandmarkExtraction(
        hand_landmarks=np.stack(hand_landmarks_seq).astype(np.float32),
        world_landmarks=np.stack(world_landmarks_seq).astype(np.float32),
        presence=np.stack(presence_seq).astype(np.uint8),
        timestamps_sec=np.array(timestamps_seq, dtype=np.float32),
    )
