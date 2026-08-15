"""Parsing puro de nomes de arquivo do dataset Kaggle Hand Wash - sem I/O.

Padrao: HandWash_{subjectID:03d}_A_{stepCode:02d}_G{_}{groupNum:02d}.mp4
Atencao: a pasta Step_1 usa "G01" (sem underscore); as outras 11 pastas usam
"G_01" (com underscore) - o regex abaixo trata os dois casos.
"""
import re
from dataclasses import dataclass

_FILENAME_PATTERN = re.compile(
    r"^HandWash_(?P<subject_id>\d{3})_A_(?P<step_code>\d{2})_G_?(?P<group_num>\d{2})\.mp4$"
)


@dataclass(frozen=True)
class ParsedVideoFilename:
    subject_id: int
    step_code: int
    raw_group_num: int  # numero do G dentro do nome do arquivo


def parse_video_filename(filename: str) -> ParsedVideoFilename:
    match = _FILENAME_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Nome de arquivo fora do padrao esperado: {filename!r}")
    return ParsedVideoFilename(
        subject_id=int(match.group("subject_id")),
        step_code=int(match.group("step_code")),
        raw_group_num=int(match.group("group_num")),
    )
