"""Adapter de I/O: construcao dos modelos Keras (arquitetura fixada na Secao 5 do plano).

Fica em infrastructure, nao domain, porque depende diretamente do framework TensorFlow/
Keras - e so uma questao de qual biblioteca ML monta o grafo, nao logica de negocio.

Opcao A e a arquitetura principal. Opcao C e o fallback especifico pra quando a Opcao A
nao converte pra TFLite builtin-only nem com GRU(unroll=True) - ver
application.check_tflite_viability (Secao 7.1 do plano). Opcao B (fallback de
overfitting, nao de compatibilidade TFLite) fica pra quando o treino de verdade
(script 06) precisar dela - nao ha uso pra ela ainda.
"""
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from handwash.config.constants import CLASSES, WINDOW_FEATURE_DIM, WINDOW_STEPS

_L2 = 1e-4


def build_primary_model(gru_unroll: bool = False) -> tf.keras.Model:
    """Opcao A - Conv1D+GRU, ~89K parametros. `gru_unroll` existe especificamente pro
    fallback de conversao TFLite da Secao 7.1: se o GRU dinamico (unroll=False) nao
    converter builtin-only, tentar de novo com unroll=True antes de recorrer a Opcao C."""
    inputs = layers.Input(shape=(WINDOW_STEPS, WINDOW_FEATURE_DIM))
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.LayerNormalization()(x)  # nao BatchNorm - evita descompasso treino/held-out
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64, return_sequences=False, kernel_regularizer=regularizers.l2(_L2), unroll=gru_unroll)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(_L2))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="handwash_opcao_a_conv_gru")


def build_fallback_c_model() -> tf.keras.Model:
    """Opcao C - so convolucional, ~84K parametros, zero risco de operacao nao suportada
    em TFLite (sem GRU/recorrencia) - fallback definitivo da Secao 7.1."""
    inputs = layers.Input(shape=(WINDOW_STEPS, WINDOW_FEATURE_DIM))
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="handwash_opcao_c_conv_only")
