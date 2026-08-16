"""Adapter de I/O: conversao Keras -> TFLite e execucao rapida via tf.lite.Interpreter.

So a parte "builtin ops" da Secao 7.1 do plano. Quantizacao INT8 + representative
dataset (Secao 7.2) e escopo do script de export de verdade (Dia 2, depois do treino
real) - aqui a conversao e sempre float32, sobre pesos aleatorios/nao treinados, porque
o unico objetivo e confirmar COMPATIBILIDADE DE OPERACAO antes de investir em treino.

Confirmado empiricamente (nao só assumido) contra tensorflow==2.21.0/keras==3.15.1: o
caminho direto `TFLiteConverter.from_keras_model()` funciona sem precisar de
export()/SavedModel intermediario - a unica coisa que decide sucesso/falha e a
configuracao do GRU (ver infrastructure.model_factory).
"""
import numpy as np
import tensorflow as tf


def convert_to_tflite_builtin_only(model: tf.keras.Model) -> bytes:
    """Converte sem SELECT_TF_OPS/Flex - so builtin ops (Secao 7.1: Flex e "ultimo
    recurso", evitado aqui de proposito). Explicito em vez de confiar no default do
    conversor, pra nao depender de um comportamento implicito que pode mudar entre
    versoes do TF."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def run_tflite_inference(tflite_bytes: bytes, input_array: np.ndarray) -> np.ndarray:
    """Roda uma unica inferencia no interpretador TFLite - confirma que o .tflite
    convertido nao so existe, mas executa de ponta a ponta e produz saida com a shape
    esperada. Le o dtype esperado direto do interpreter em vez de assumir float32, pra
    continuar correto quando um modelo INT8 (Secao 7.2, fora de escopo hoje) passar
    por aqui no futuro."""
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_detail["index"], input_array.astype(input_detail["dtype"]))
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"])
