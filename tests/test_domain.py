"""Testes da camada domain - logica pura, sem I/O, sem mocks necessarios."""
import numpy as np
import pytest

from handwash.config.constants import CLASSES
from handwash.domain.class_scheme import class_to_index, raw_folder_to_class
from handwash.domain.filename_parsing import parse_video_filename
from handwash.domain.gap_filling import (
    MAX_SHORT_GAP_FRAMES,
    at_least_one_hand_rate,
    both_hands_rate,
    detection_rate,
    interpolate_short_gaps,
)
from handwash.domain.sampling import rotated_subject_sample, sample_subjects_for_folder
from handwash.domain.splits import split_for_subject, validate_split_integrity


def test_parse_filename_with_underscore():
    parsed = parse_video_filename("HandWash_018_A_11_G_05.mp4")
    assert parsed.subject_id == 18
    assert parsed.step_code == 11


def test_parse_filename_without_underscore_step1_quirk():
    parsed = parse_video_filename("HandWash_001_A_01_G01.mp4")
    assert parsed.subject_id == 1
    assert parsed.step_code == 1


def test_parse_filename_rejects_garbage():
    with pytest.raises(ValueError):
        parse_video_filename("nao_e_um_nome_valido.mp4")


def test_class_scheme_merges_left_right():
    assert raw_folder_to_class("Step_2_Left") == raw_folder_to_class("Step_2_Right")
    assert raw_folder_to_class("Step_1") != raw_folder_to_class("Step_3")


def test_class_scheme_unknown_folder_raises():
    with pytest.raises(ValueError):
        raw_folder_to_class("Step_99_Nonexistent")


def test_class_to_index_unknown_class_raises():
    with pytest.raises(ValueError):
        class_to_index("classe_que_nao_existe")


def test_class_scheme_wrist_is_its_own_class_not_other():
    wrist_class = raw_folder_to_class("Step_7_Left")
    assert wrist_class == "step7_pulso"
    assert wrist_class in CLASSES


def test_class_to_index_matches_frozen_order():
    assert class_to_index(CLASSES[0]) == 0
    assert class_to_index(CLASSES[-1]) == len(CLASSES) - 1


def test_split_integrity_passes_on_current_config():
    validate_split_integrity()  # nao deve levantar excecao


def test_every_subject_has_exactly_one_split():
    for subject_id in range(1, 26):
        split = split_for_subject(subject_id)
        assert split in {"train", "val", "test"}


def test_split_for_unknown_subject_raises():
    with pytest.raises(ValueError):
        split_for_subject(999)


def test_interpolate_short_gap_fills_between_valid_frames():
    n_frames, n_slots = 5, 1
    landmarks = np.zeros((n_frames, n_slots, 21, 3), dtype=np.float32)
    presence = np.array([[1], [0], [0], [0], [1]], dtype=np.uint8)
    landmarks[0] = 0.0
    landmarks[4] = 4.0

    filled_landmarks, filled_presence = interpolate_short_gaps(landmarks, presence)

    assert filled_presence.sum() == 5
    assert np.isclose(filled_landmarks[2, 0, 0, 0], 2.0, atol=0.01)


def test_interpolate_long_gap_stays_zero_filled():
    n_frames, n_slots = 10, 1
    landmarks = np.ones((n_frames, n_slots, 21, 3), dtype=np.float32)
    presence = np.zeros((n_frames, n_slots), dtype=np.uint8)
    presence[0] = 1
    presence[9] = 1  # gap de 8 frames, maior que MAX_SHORT_GAP_FRAMES=5

    _, filled_presence = interpolate_short_gaps(landmarks, presence)

    assert filled_presence[5, 0] == 0


def test_interpolate_gap_boundary_exactly_max_is_filled():
    # exatamente MAX_SHORT_GAP_FRAMES (5) frames faltando entre duas deteccoes
    # validas - contrato do plano diz "gaps <= 5 frames: interpola".
    n_frames = MAX_SHORT_GAP_FRAMES + 2  # indices 0 (valido) .. 6 (valido), 5 faltando no meio
    landmarks = np.zeros((n_frames, 1, 21, 3), dtype=np.float32)
    presence = np.zeros((n_frames, 1), dtype=np.uint8)
    presence[0] = 1
    presence[-1] = 1

    _, filled_presence = interpolate_short_gaps(landmarks, presence)

    assert filled_presence.sum() == n_frames  # todo o gap de 5 foi preenchido


def test_interpolate_gap_boundary_one_past_max_stays_unfilled():
    # MAX_SHORT_GAP_FRAMES + 1 (6) frames faltando - um a mais que o limite,
    # nao deveria interpolar nada.
    n_frames = MAX_SHORT_GAP_FRAMES + 3  # 6 faltando no meio
    landmarks = np.zeros((n_frames, 1, 21, 3), dtype=np.float32)
    presence = np.zeros((n_frames, 1), dtype=np.uint8)
    presence[0] = 1
    presence[-1] = 1

    _, filled_presence = interpolate_short_gaps(landmarks, presence)

    assert filled_presence.sum() == 2  # so os dois frames originais, nada preenchido


def test_detection_rate_is_fraction_present():
    presence = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.uint8)
    assert detection_rate(presence) == pytest.approx(3 / 6)


def test_at_least_one_hand_rate_counts_partial_frames_as_present():
    # frame 0: so mao esquerda: conta. frame 1: as duas: conta. frame 2: nenhuma: nao conta.
    presence = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.uint8)
    assert at_least_one_hand_rate(presence) == pytest.approx(2 / 3)


def test_both_hands_rate_only_counts_frames_with_both():
    presence = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.uint8)
    assert both_hands_rate(presence) == pytest.approx(1 / 3)


def test_at_least_one_hand_rate_can_be_much_higher_than_detection_rate():
    # cenario tipico de passo com maos entrelacadas: quase sempre 1 mao detectada,
    # raramente as duas ao mesmo tempo - detection_rate cai bem abaixo de
    # at_least_one_hand_rate, o que e esperado, nao um bug.
    presence = np.array([[1, 0]] * 8 + [[1, 1]] * 2, dtype=np.uint8)
    assert at_least_one_hand_rate(presence) == pytest.approx(1.0)
    assert detection_rate(presence) == pytest.approx((8 * 1 + 2 * 2) / 20)


def test_rotated_subject_sample_wraps_around():
    ids = [10, 20, 30, 40, 50]
    assert rotated_subject_sample(ids, folder_idx=0, n=2) == [10, 20]
    assert rotated_subject_sample(ids, folder_idx=1, n=2) == [20, 30]
    assert rotated_subject_sample(ids, folder_idx=4, n=2) == [50, 10]  # da a volta
    assert rotated_subject_sample(ids, folder_idx=5, n=2) == [10, 20]  # modulo 5, volta ao inicio


def test_rotated_subject_sample_empty_input():
    assert rotated_subject_sample([], folder_idx=3, n=2) == []


def test_sample_subjects_g05_pick_varies_across_folders():
    # regressao direta do bug encontrado em revisao de codigo: a versao antiga
    # rotacionava a lista inteira de 25 sujeitos por folder_idx (0-11), o que
    # nunca deslocava o suficiente pra alcancar o subgrupo G05 de forma
    # diferente - o sujeito escolhido do G05 era sempre o mesmo (18) em toda
    # pasta. Girar cada subgrupo de forma independente corrige isso.
    non_g05 = list(range(1, 18))
    g05 = list(range(18, 26))
    g05_picks = {sample_subjects_for_folder(i, non_g05, g05, 4)[-1] for i in range(12)}
    assert len(g05_picks) > 1


def test_sample_subjects_always_includes_one_g05():
    non_g05 = list(range(1, 18))
    g05 = list(range(18, 26))
    for folder_idx in range(12):
        chosen = sample_subjects_for_folder(folder_idx, non_g05, g05, 4)
        assert any(sid in g05 for sid in chosen)


def test_sample_subjects_respects_n_per_folder():
    non_g05 = list(range(1, 18))
    g05 = list(range(18, 26))
    for n in (1, 3, 4, 6):
        chosen = sample_subjects_for_folder(0, non_g05, g05, n)
        assert len(chosen) == n
