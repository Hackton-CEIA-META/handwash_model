"""Split treino/val/teste por sujeito, ciente de G-group (Secao 3 do plano).

Regra dura: toda janela derivada dos videos do sujeito X vai inteiramente para
UM split so. validate_split_integrity() garante isso no nivel de configuracao,
antes de qualquer dado ser processado.
"""
from handwash.config.constants import SUBJECT_SPLITS

_ALL_SUBJECT_IDS = range(1, 26)


def split_for_subject(subject_id: int) -> str:
    for split_name, subject_ids in SUBJECT_SPLITS.items():
        if subject_id in subject_ids:
            return split_name
    raise ValueError(f"Sujeito {subject_id} nao esta em nenhum split - SUBJECT_SPLITS esta incompleto?")


def validate_split_integrity() -> None:
    """Garante que nenhum sujeito aparece em mais de um split e que todos os 25 estao cobertos."""
    all_ids = []
    for subject_ids in SUBJECT_SPLITS.values():
        all_ids.extend(subject_ids)

    duplicates = {sid for sid in all_ids if all_ids.count(sid) > 1}
    if duplicates:
        raise ValueError(f"Sujeitos aparecem em mais de um split: {sorted(duplicates)}")

    expected = set(_ALL_SUBJECT_IDS)
    missing = expected - set(all_ids)
    if missing:
        raise ValueError(f"Sujeitos ausentes de qualquer split: {sorted(missing)}")

    extra = set(all_ids) - expected
    if extra:
        raise ValueError(f"IDs de sujeito fora do intervalo esperado (1-25): {sorted(extra)}")
