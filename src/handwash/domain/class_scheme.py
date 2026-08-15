"""Mapeamento de pasta bruta -> classe final.

Decisao (Secao 2 do plano): mesclar Left/Right em 7 classes finais, mantendo o
passo do pulso (Step_7) como classe propria - nao um bucket generico "Other"
como o repositorio de referencia edi-riga/handwash faz.
"""
from handwash.config.constants import RAW_FOLDER_TO_CLASS, CLASSES


def raw_folder_to_class(raw_folder_name: str) -> str:
    try:
        return RAW_FOLDER_TO_CLASS[raw_folder_name]
    except KeyError as exc:
        raise ValueError(f"Pasta bruta desconhecida, sem classe mapeada: {raw_folder_name!r}") from exc


def class_to_index(class_name: str) -> int:
    return CLASSES.index(class_name)
