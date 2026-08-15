"""Amostragem deterministica de sujeitos para checagens de viabilidade - logica pura.

Extraido para domain apos revisao de codigo encontrar um bug real na versao
anterior (vivia como funcao privada dentro de application, sem teste): rotacionar
a lista inteira de 25 sujeitos por folder_idx (0-11, uma por pasta bruta) nunca
desloca o suficiente pra alcancar o subgrupo G05 (indices 17-24) de forma
diferente - o "primeiro" sujeito do G05 amostrado era sempre o 18, nunca
variava. A correcao gira cada subgrupo (nao-G05 e G05) de forma independente,
cada um modulo o proprio tamanho.
"""


def rotated_subject_sample(subject_ids: list[int], folder_idx: int, n: int) -> list[int]:
    """Retorna os `n` primeiros IDs de `subject_ids`, girados por `folder_idx`
    posicoes (modulo o tamanho da lista) - garante cobertura variada ao longo
    de chamadas com folder_idx crescente, mesmo quando `subject_ids` e curta."""
    if not subject_ids or n <= 0:
        return []
    offset = folder_idx % len(subject_ids)
    rotated = subject_ids[offset:] + subject_ids[:offset]
    return rotated[:n]


def sample_subjects_for_folder(
    folder_idx: int, non_g05_ids: list[int], g05_ids: list[int], n_per_folder: int
) -> list[int]:
    """Amostra sujeitos pra checagem de viabilidade numa pasta bruta, girando
    non_g05_ids e g05_ids de forma INDEPENDENTE por folder_idx - garante que o
    G05 (setup de camera visivelmente diferente, 1920x1080) tenha diversidade
    real de sujeitos amostrados entre pastas, nao sempre o mesmo."""
    n_non_g05 = max(n_per_folder - 1, 1)
    chosen = rotated_subject_sample(non_g05_ids, folder_idx, n_non_g05)
    chosen += rotated_subject_sample(g05_ids, folder_idx, 1)
    return chosen[:n_per_folder]
