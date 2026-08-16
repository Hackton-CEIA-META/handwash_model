"""Testes da camada domain - logica pura, sem I/O, sem mocks necessarios."""
import numpy as np
import pytest

from handwash.config.constants import CLASSES
from handwash.domain.augmentation import (
    augment_window,
    jitter_time,
    landmark_noise,
    mirror_lr,
    rotate_z,
    scale_jitter,
)
from handwash.domain.class_scheme import class_to_index, raw_folder_to_class
from handwash.domain.class_weights import compute_class_weights
from handwash.domain.filename_parsing import parse_video_filename
from handwash.domain.gap_filling import (
    MAX_SHORT_GAP_FRAMES,
    at_least_one_hand_rate,
    both_hands_rate,
    detection_rate,
    interpolate_short_gaps,
)
from handwash.domain.normalization import normalize_and_mask, normalize_hand, zero_out_absent_hands
from handwash.domain.resampling import resample_linear, resample_nearest, target_grid
from handwash.domain.sampling import rotated_subject_sample, sample_subjects_for_folder
from handwash.domain.splits import (
    assert_no_cross_split_subjects,
    split_for_subject,
    validate_split_integrity,
)
from handwash.domain.windowing import slice_windows, window_start_indices


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


# --- resampling (Dia 2, Secao 4 do plano: 30fps nativo -> 15fps por timestamp) ---

def test_target_grid_spacing_and_count():
    grid = target_grid(duration_sec=0.3, target_fps=10.0)
    assert np.allclose(grid, [0.0, 0.1, 0.2, 0.3], atol=1e-6)


def test_resample_linear_interpolates_exactly_on_a_linear_ramp():
    timestamps = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    values = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    target_ts = np.array([0.05, 0.15, 0.25], dtype=np.float32)

    resampled = resample_linear(values, timestamps, target_ts)

    assert np.allclose(resampled, [[5.0], [15.0], [25.0]], atol=1e-4)


def test_resample_linear_clamps_outside_native_range_no_extrapolation():
    timestamps = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    values = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    target_ts = np.array([-1.0, 5.0], dtype=np.float32)  # bem fora do intervalo nativo

    resampled = resample_linear(values, timestamps, target_ts)

    assert np.allclose(resampled, [[0.0], [20.0]])  # satura nas pontas, nao extrapola


def test_resample_nearest_picks_closer_native_frame_not_a_blend():
    timestamps = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    presence = np.array([[1], [0], [1], [0]], dtype=np.uint8)
    target_ts = np.array([0.03, 0.08, 0.24], dtype=np.float32)

    resampled = resample_nearest(presence, timestamps, target_ts)

    # 0.03 -> mais perto de 0.0 (presence=1); 0.08 -> mais perto de 0.1 (presence=0);
    # 0.24 -> mais perto de 0.2 (presence=1). Nunca um valor fracionario.
    assert resampled.flatten().tolist() == [1, 0, 1]


# --- windowing (Dia 2, Secao 4 do plano: janelas de 30 passos, sem padding) ---

def test_window_start_indices_basic_stride():
    assert window_start_indices(n_frames=10, window_steps=3, stride_steps=2) == [0, 2, 4, 6]


def test_window_start_indices_drops_incomplete_tail_no_padding():
    # n_frames=5, window=3, stride=1 caberia janelas em 0,1,2 - nao ha janela em 3
    # (cobriria 3,4,5 mas so existem indices ate 4) nem em 4.
    assert window_start_indices(n_frames=5, window_steps=3, stride_steps=1) == [0, 1, 2]


def test_window_start_indices_shorter_than_window_returns_empty():
    assert window_start_indices(n_frames=2, window_steps=3, stride_steps=1) == []


def test_slice_windows_preserves_temporal_order_and_shape():
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    starts = [0, 2, 4]

    windows = slice_windows(features, starts, window_steps=3)

    assert windows.shape == (3, 3, 2)
    assert np.array_equal(windows[0], features[0:3])
    assert np.array_equal(windows[1], features[2:5])
    assert np.array_equal(windows[2], features[4:7])


def test_slice_windows_empty_starts_returns_correctly_shaped_empty_array():
    features = np.zeros((10, 2), dtype=np.float32)
    windows = slice_windows(features, [], window_steps=3)
    assert windows.shape == (0, 3, 2)


# --- normalization (Dia 2, Secao 5 do plano: origem no pulso, escala pulso->MCP-medio) ---

def test_normalize_hand_wrist_maps_to_origin_and_mcp_distance_becomes_one():
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [1.0, 1.0, 1.0]  # pulso
    landmarks[9] = [1.0, 1.0, 3.0]  # MCP do dedo medio - distancia 2.0 do pulso
    landmarks[5] = [3.0, 1.0, 1.0]  # ponto arbitrario

    normalized = normalize_hand(landmarks)

    assert np.allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-6)
    assert np.isclose(np.linalg.norm(normalized[9]), 1.0, atol=1e-6)
    assert np.allclose(normalized[5], [1.0, 0.0, 0.0], atol=1e-6)


def test_normalize_hand_zero_filled_frame_stays_zero_no_nan_or_inf():
    landmarks = np.zeros((21, 3), dtype=np.float32)  # simula presence=0
    normalized = normalize_hand(landmarks)
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized, 0.0)


def test_normalize_hand_clips_extreme_scale_outliers():
    # Regressao do achado real ao investigar a queda de acuracia na quantizacao INT8: uma
    # mao mal rastreada mas nao ausente (presence=1) pode ter distancia pulso->MCP minuscula
    # sem bater o piso _MIN_SCALE, fazendo a divisao explodir. Sem clip, isso destroi a
    # calibracao MinMax do TFLite mesmo sendo raro (~0.0004% dos valores reais).
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.0, 0.0, 0.0]  # pulso
    landmarks[9] = [0.001, 0.0, 0.0]  # MCP quase colado no pulso - escala minuscula, nao zero
    landmarks[5] = [1.0, 0.0, 0.0]  # ponto normal, mas a distancia do pulso vira gigante apos dividir

    normalized = normalize_hand(landmarks)

    assert np.all(np.abs(normalized) <= 5.0 + 1e-6)
    assert np.all(np.isfinite(normalized))


def test_normalize_hand_broadcasts_independently_over_leading_batch_dims():
    landmarks = np.zeros((2, 2, 21, 3), dtype=np.float32)  # (frames, slots, 21, 3)
    landmarks[0, 0, 0] = [0.0, 0.0, 0.0]
    landmarks[0, 0, 9] = [0.0, 0.0, 1.0]  # escala 1.0 nesta mao
    landmarks[1, 1, 0] = [5.0, 5.0, 5.0]
    landmarks[1, 1, 9] = [5.0, 5.0, 9.0]  # escala 4.0 nesta outra mao/frame

    normalized = normalize_hand(landmarks)

    assert np.isclose(np.linalg.norm(normalized[0, 0, 9]), 1.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(normalized[1, 1, 9]), 1.0, atol=1e-6)


def test_zero_out_absent_hands_clears_only_frames_marked_absent():
    normalized = np.ones((3, 2, 21, 3), dtype=np.float32)
    presence = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)

    result = zero_out_absent_hands(normalized, presence)

    assert np.allclose(result[0, 1], 0.0)  # ausente
    assert np.allclose(result[1, 0], 0.0)  # ausente
    assert np.allclose(result[0, 0], 1.0)  # presente, inalterado
    assert np.allclose(result[2, 0], 1.0)
    assert np.allclose(result[2, 1], 1.0)
    assert np.allclose(normalized, 1.0)  # nao muta a entrada original


def test_composed_pipeline_gap_boundary_leak_is_closed_by_normalize_and_mask():
    # Regressao do achado de revisao de codigo: um frame reamostrado bem na fronteira
    # entre deteccao real (frame 0) e um gap longo zero-preenchido (frame 1) interpola
    # uma pose ENCOLHIDA em direcao a zero. Sem mascarar por presence, normalize_hand()
    # (invariante a escala) reconstroi a pose real mesmo com presence=0 - o teste abaixo
    # prova numericamente as duas metades: (1) o vazamento existe se so normalize_hand()
    # for chamada, (2) normalize_and_mask() fecha ele de fato.
    native_ts = np.array([0.0, 0.1], dtype=np.float32)
    landmarks = np.zeros((2, 1, 21, 3), dtype=np.float32)
    landmarks[0, 0, 0] = [1.0, 1.0, 1.0]  # pulso, frame 0 (real)
    landmarks[0, 0, 9] = [1.0, 1.0, 3.0]  # MCP, frame 0 (real) - escala 2.0
    landmarks[0, 0, 5] = [3.0, 1.0, 1.0]  # ponto arbitrario, frame 0 (real)
    # frame 1 fica zero (inicio de um gap longo zero-preenchido), presence=0
    presence = np.array([[1], [0]], dtype=np.uint8)

    target_ts = np.array([0.07], dtype=np.float32)  # alpha=0.7 - puxa pro lado zero-preenchido

    resampled_landmarks = resample_linear(landmarks, native_ts, target_ts)
    resampled_presence = resample_nearest(presence, native_ts, target_ts)
    assert resampled_presence[0, 0] == 0  # nearest escolhe o frame ausente (alpha=0.7 >= 0.5)

    unmasked = normalize_hand(resampled_landmarks)
    assert np.allclose(unmasked[0, 0, 5], [1.0, 0.0, 0.0], atol=1e-5)  # o vazamento: pose real "vaza"

    fixed = normalize_and_mask(resampled_landmarks, resampled_presence)
    assert np.allclose(fixed[0, 0], 0.0)  # normalize_and_mask fecha o vazamento de fato


# --- split leakage assertion contra dados montados (Dia 2, Secao 3 do plano) ---

def test_assert_no_cross_split_subjects_passes_when_consistent():
    rows = [
        {"subject_id": 1, "split": "train"},  # 1 esta em SUBJECT_SPLITS["train"]
        {"subject_id": 1, "split": "train"},
        {"subject_id": 4, "split": "val"},  # 4 esta em SUBJECT_SPLITS["val"]
    ]
    assert_no_cross_split_subjects(rows)  # nao deve levantar excecao


def test_assert_no_cross_split_subjects_raises_on_leakage():
    rows = [
        {"subject_id": 1, "split": "train"},
        {"subject_id": 1, "split": "test"},
    ]
    with pytest.raises(ValueError):
        assert_no_cross_split_subjects(rows)


def test_assert_no_cross_split_subjects_catches_staleness_vs_subject_splits():
    # Regressao do achado de revisao de codigo: a versao antiga so checava consistencia
    # ENTRE as linhas dadas, nunca contra o SUBJECT_SPLITS atual - um manifesto gerado
    # sob um split antigo, mas internamente autoconsistente, passava batido. Aqui so ha
    # UMA linha (autoconsistente por definicao) com um split que nao bate com o real.
    rows = [{"subject_id": 1, "split": "val"}]  # 1 e train de verdade (SUBJECT_SPLITS)
    with pytest.raises(ValueError):
        assert_no_cross_split_subjects(rows)


# --- augmentacao (Dia 2, Secao 5 do plano - so treino, nunca val/teste) ---

def _make_window(landmarks, presence):
    """landmarks: (T,2,21,3), presence: (T,2). Monta uma janela (T,128) igual ao
    contrato do gold layer, sem depender de helpers privados do modulo testado."""
    flat = landmarks.reshape(landmarks.shape[0], -1)
    return np.concatenate([flat, presence.astype(np.float32)], axis=-1).astype(np.float32)


def test_mirror_lr_swaps_slots_and_flips_x():
    landmarks = np.zeros((2, 2, 21, 3), dtype=np.float32)
    landmarks[:, 0, 5] = [1.0, 2.0, 3.0]  # slot esquerdo, ponto 5
    landmarks[:, 1, 5] = [4.0, 5.0, 6.0]  # slot direito, ponto 5
    presence = np.array([[1, 0], [0, 1]], dtype=np.float32)
    window = _make_window(landmarks, presence)

    mirrored = mirror_lr(window)
    m_landmarks = mirrored[:, :126].reshape(2, 2, 21, 3)
    m_presence = mirrored[:, 126:]

    assert np.allclose(m_landmarks[:, 0, 5], [-4.0, 5.0, 6.0])  # direito virou esquerdo, x invertido
    assert np.allclose(m_landmarks[:, 1, 5], [-1.0, 2.0, 3.0])  # esquerdo virou direito, x invertido
    assert np.array_equal(m_presence, presence[:, ::-1])


def test_mirror_lr_preserves_zero_for_absent_hand():
    landmarks = np.zeros((3, 2, 21, 3), dtype=np.float32)
    presence = np.zeros((3, 2), dtype=np.float32)
    window = _make_window(landmarks, presence)
    assert np.allclose(mirror_lr(window), 0.0)


def test_rotate_z_preserves_xy_norm_and_absent_hand_zero():
    rng = np.random.default_rng(0)
    landmarks = np.zeros((1, 2, 21, 3), dtype=np.float32)
    landmarks[0, 0, 5] = [3.0, 4.0, 1.0]  # norma no plano xy = 5.0
    presence = np.array([[1, 0]], dtype=np.float32)  # slot 1 ausente
    window = _make_window(landmarks, presence)

    rotated = rotate_z(window, max_degrees=15.0, rng=rng)
    r_landmarks = rotated[:, :126].reshape(1, 2, 21, 3)

    assert np.isclose(np.linalg.norm(r_landmarks[0, 0, 5, :2]), 5.0, atol=1e-5)  # rotacao preserva norma
    assert np.isclose(r_landmarks[0, 0, 5, 2], 1.0)  # z inalterado
    assert np.allclose(r_landmarks[0, 1], 0.0)  # slot ausente continua zero


def test_scale_jitter_scales_and_preserves_absent_hand_zero():
    rng = np.random.default_rng(0)
    landmarks = np.zeros((1, 2, 21, 3), dtype=np.float32)
    landmarks[0, 0, 5] = [2.0, 0.0, 0.0]
    presence = np.array([[1, 0]], dtype=np.float32)
    window = _make_window(landmarks, presence)

    scaled = scale_jitter(window, scale_range=(2.0, 2.0), rng=rng)  # fator fixo = 2.0
    s_landmarks = scaled[:, :126].reshape(1, 2, 21, 3)

    assert np.allclose(s_landmarks[0, 0, 5], [4.0, 0.0, 0.0])
    assert np.allclose(s_landmarks[0, 1], 0.0)


def test_landmark_noise_perturbs_present_but_zeroes_absent():
    rng = np.random.default_rng(0)
    landmarks = np.zeros((1, 2, 21, 3), dtype=np.float32)
    landmarks[0, 0, 5] = [1.0, 1.0, 1.0]
    presence = np.array([[1, 0]], dtype=np.float32)
    window = _make_window(landmarks, presence)

    noisy = landmark_noise(window, sigma=0.1, rng=rng)
    n_landmarks = noisy[:, :126].reshape(1, 2, 21, 3)

    assert not np.allclose(n_landmarks[0, 0, 5], [1.0, 1.0, 1.0])  # ruido de fato mudou o presente
    assert np.allclose(n_landmarks[0, 1], 0.0)  # ausente continua exatamente zero, sem ruido


def test_jitter_time_preserves_shape_and_boundary_absence_stays_zero():
    rng = np.random.default_rng(0)
    landmarks = np.zeros((4, 2, 21, 3), dtype=np.float32)
    landmarks[0:2, 0, 5] = [1.0, 1.0, 1.0]  # frames 0-1, slot 0: mao real presente
    presence = np.zeros((4, 2), dtype=np.float32)
    presence[0:2, 0] = 1.0  # slot 1 sempre ausente; slot 0 vira ausente nos frames 2-3
    window = _make_window(landmarks, presence)

    shifted = jitter_time(window, max_shift_steps=0.5, rng=rng)
    assert shifted.shape == window.shape
    assert np.all(np.isfinite(shifted))

    s_landmarks = shifted[:, :126].reshape(4, 2, 21, 3)
    s_presence = shifted[:, 126:]
    assert np.allclose(s_landmarks[s_presence == 0], 0.0)  # nunca vaza pose onde presence=0


def test_augment_window_preserves_shape_and_finiteness():
    rng = np.random.default_rng(0)
    landmarks = np.random.default_rng(1).normal(size=(30, 2, 21, 3)).astype(np.float32)
    presence = np.ones((30, 2), dtype=np.float32)
    window = _make_window(landmarks, presence)

    for _ in range(10):  # varias chamadas, cobre combinacoes diferentes de augmentations
        augmented = augment_window(window, rng)
        assert augmented.shape == window.shape
        assert np.all(np.isfinite(augmented))


# --- pesos de classe (Dia 2, Secao 6 do plano) ---

def test_compute_class_weights_balanced_case_gives_equal_weights():
    class_names = [CLASSES[0]] * 10 + [CLASSES[1]] * 10
    weights = compute_class_weights(class_names)
    assert np.isclose(weights[0], weights[1])


def test_compute_class_weights_matches_known_25_vs_50_ratio():
    # replica o desbalanceamento real do dataset (Secao 1/Labels do CLAUDE.md): 2 classes
    # com 25 videos, 5 classes com 50.
    class_names = [CLASSES[0]] * 25 + [CLASSES[2]] * 25
    for name in (CLASSES[1], CLASSES[3], CLASSES[4], CLASSES[5], CLASSES[6]):
        class_names += [name] * 50
    weights = compute_class_weights(class_names)

    assert np.isclose(weights[0] / weights[1], 2.0, atol=1e-6)  # classe de 25 pesa 2x mais
    assert np.isclose(weights[0], 300 / (7 * 25))


def test_compute_class_weights_omits_absent_class():
    class_names = [CLASSES[0]] * 5  # so uma classe presente
    weights = compute_class_weights(class_names)
    assert set(weights.keys()) == {0}
