"""Pesos de classe (Secao 6 do plano): calculados do zero a partir de uma lista de
rotulos (o chamador le `data/bronze/manifest.csv` e passa a coluna `merged_class` - a
mesma unidade, VIDEO, que CLAUDE.md usa pra descrever o desbalanceamento: "exatamente 25
vs 50 videos"). NAO reaproveitar o `get_weights_dict()` do edi-riga, que indexa por
posicao numerica (`str(i)`) e quebra silenciosamente com nomes de classe descritivos
como os nossos (Secao 6 do plano).
"""
from collections import Counter

from handwash.config.constants import CLASSES


def compute_class_weights(class_names: list[str]) -> dict[int, float]:
    """class_names: uma entrada por exemplo (video ou janela - decisao de quem chama).
    Retorna {class_idx: peso}, esquema "balanced" padrao: peso = n_total/(n_classes*count).
    Peso 1.0 = classe na proporcao media; >1.0 = sub-representada; <1.0 = sobre-representada.
    """
    counts = Counter(class_names)
    n_total = len(class_names)
    n_classes = len(CLASSES)
    return {
        CLASSES.index(name): n_total / (n_classes * counts[name])
        for name in CLASSES
        if counts.get(name, 0) > 0
    }
