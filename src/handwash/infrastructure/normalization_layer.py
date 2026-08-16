"""Adapter de I/O: domain.normalization.normalize_and_mask reimplementada em ops de TF,
pra ficar embutida no grafo exportado (CLAUDE.md: "normalizacao... assada no grafo
exportado, nao reimplementada em Kotlin").

Confirmado empiricamente antes de usar em producao (mesma disciplina do gate TFLite e do
probe de treino): a saida desta camada bate EXATAMENTE (diff maximo 0.0) com a versao
numpy sobre dado aleatorio, e o grafo composto (esta camada + Conv1D+GRU(unroll=True)+
Dense) converte builtin-only sem erro.
"""
import tensorflow as tf

from handwash.config.constants import MIDDLE_MCP_LANDMARK_IDX, NUM_LANDMARK_COORDS, WRIST_LANDMARK_IDX
from handwash.infrastructure.landmark_extractor import NUM_COORDS, NUM_HAND_SLOTS, NUM_LANDMARKS

_MIN_SCALE = 0.02  # mesmo piso de domain.normalization - ver nota la (seguranca sob quantizacao INT8)
_CLIP_ABS_VALUE = 5.0  # mesmo clip de domain.normalization.normalize_hand - ver nota la


class NormalizeAndMaskLayer(tf.keras.layers.Layer):
    """Entrada: (batch, T, 128) = 126 coords de landmark BRUTOS (2 maos x 21 pontos x 3D,
    mesmo layout do gold layer, so que sem normalizar) + 2 flags de presenca. Sem pesos
    treinaveis - e pura transformacao geometrica (Secao 5 do plano), nao aprendida."""

    def call(self, inputs):
        landmarks = inputs[..., :NUM_LANDMARK_COORDS]
        presence = inputs[..., NUM_LANDMARK_COORDS:]
        shape = tf.shape(inputs)
        landmarks = tf.reshape(landmarks, (shape[0], shape[1], NUM_HAND_SLOTS, NUM_LANDMARKS, NUM_COORDS))

        wrist = landmarks[..., WRIST_LANDMARK_IDX, :]
        middle_mcp = landmarks[..., MIDDLE_MCP_LANDMARK_IDX, :]
        scale = tf.maximum(tf.norm(middle_mcp - wrist, axis=-1), _MIN_SCALE)

        centered = landmarks - wrist[..., tf.newaxis, :]
        normalized = centered / scale[..., tf.newaxis, tf.newaxis]
        normalized = tf.clip_by_value(normalized, -_CLIP_ABS_VALUE, _CLIP_ABS_VALUE)
        masked = normalized * presence[..., tf.newaxis, tf.newaxis]  # presence e 0/1 - zera mao ausente

        flat = tf.reshape(masked, (shape[0], shape[1], NUM_LANDMARK_COORDS))
        return tf.concat([flat, presence], axis=-1)
