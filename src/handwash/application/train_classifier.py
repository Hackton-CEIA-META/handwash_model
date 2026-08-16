"""Caso de uso: treino do classificador (Secao 5/6 do plano).

Arquitetura: Opcao A vencedora do gate de viabilidade TFLite (GRU unroll=True - ver
scripts/smoke_test_tflite.py e infrastructure.model_factory.build_primary_model). Class
weights calculados do manifest bronze (nao da posicional do edi-riga). Augmentation
on-the-fly SO no treino (domain.augmentation), um sorteio novo por epoca via
tf.keras.utils.Sequence - val/teste sempre limpos, sem augmentation.

Nota de API confirmada empiricamente (nao assumida): `SparseCategoricalCrossentropy`
nesta versao de Keras (3.15) NAO aceita `label_smoothing` - so `CategoricalCrossentropy`
aceita, e exige alvo one-hot. Por isso os rotulos sao one-hot-encoded antes do treino;
a avaliacao (matriz de confusao, acuracia por classe/video) continua comparando indices
inteiros diretamente, que e o formato que sai de `data/gold/{split}/windows.npz`.
"""
import csv
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers, utils

from handwash.config.constants import CLASSES
from handwash.config.paths import BRONZE_MANIFEST, CHECKPOINTS_DIR, GOLD_DIR
from handwash.domain.augmentation import augment_window
from handwash.domain.class_weights import compute_class_weights
from handwash.infrastructure.model_factory import build_primary_model

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 4
HARD_FLOOR_ACCURACY = 0.7511  # Lulla et al. 2021 - Secao 6 do plano
STRETCH_REFERENCE_ACCURACY = 0.935  # GhostHandEgoNet no Kaggle - referencia, nao meta obrigatoria

CHECKPOINT_PATH = CHECKPOINTS_DIR / "handwash_classifier_best.keras"
TRAINING_REPORT_PATH = CHECKPOINTS_DIR / "training_report.json"


class _AugmentedTrainSequence(utils.PyDataset):
    """Gera batches de treino com augmentation on-the-fly (Secao 5, item 5 do plano) -
    um sorteio aleatorio novo por amostra a cada epoca (via on_epoch_end), nao um
    conjunto augmentado fixo salvo em disco. X_clean/y_onehot nunca sao alterados -
    cada __getitem__ parte deles e monta uma copia augmentada so daquele batch."""

    def __init__(self, X_clean, y_onehot, batch_size, seed, **kwargs):
        super().__init__(**kwargs)
        self.X_clean = X_clean
        self.y_onehot = y_onehot
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(X_clean))

    def __len__(self):
        return int(np.ceil(len(self.X_clean) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = np.stack([augment_window(self.X_clean[i], self.rng) for i in batch_idx])
        y_batch = self.y_onehot[batch_idx]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.rng.shuffle(self.indices)


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(GOLD_DIR / split / "windows.npz") as data:
        return data["X"], data["y"]


def _class_names_from_bronze_manifest() -> list[str]:
    with BRONZE_MANIFEST.open(newline="", encoding="utf-8") as f:
        return [row["merged_class"] for row in csv.DictReader(f)]


def _build_and_compile_model() -> tf.keras.Model:
    model = build_primary_model(gru_unroll=True)  # Opcao A, vencedora confirmada do gate TFLite
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model


def _evaluate(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    overall_accuracy = float(np.mean(preds == y_test))

    per_class_accuracy = {}
    for class_idx, class_name in enumerate(CLASSES):
        mask = y_test == class_idx
        if mask.sum() > 0:
            per_class_accuracy[class_name] = float(np.mean(preds[mask] == y_test[mask]))

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for true_idx, pred_idx in zip(y_test, preds):
        confusion[true_idx, pred_idx] += 1

    correct_mask = preds == y_test
    calibration = {
        "mean_confidence_correct": float(confidences[correct_mask].mean()) if correct_mask.any() else None,
        "mean_confidence_incorrect": float(confidences[~correct_mask].mean()) if (~correct_mask).any() else None,
    }

    return {
        "test_accuracy_overall": overall_accuracy,
        "test_accuracy_per_class": per_class_accuracy,
        "confusion_matrix": {"classes": CLASSES, "matrix": confusion.tolist()},
        "calibration": calibration,
        **_per_video_majority_vote_accuracy(preds),
    }


def _per_video_majority_vote_accuracy(preds: np.ndarray) -> dict:
    """Junta as predicoes por janela de volta ao video-fonte via split_row_idx
    (data/gold/manifest.csv), vota por maioria entre as janelas de cada video e compara
    contra a classe verdadeira - metrica mais proxima de "o sistema acertou essa
    higienizacao" que a acuracia por janela isolada (Secao 6 do plano)."""
    with (GOLD_DIR / "manifest.csv").open(newline="", encoding="utf-8") as f:
        test_rows = [row for row in csv.DictReader(f) if row["split"] == "test"]
    test_rows.sort(key=lambda row: int(row["split_row_idx"]))  # bate com a ordem de X_test/y_test

    by_video: dict[str, dict] = {}
    for row, pred in zip(test_rows, preds):
        entry = by_video.setdefault(row["video_stem"], {"true_idx": int(row["class_idx"]), "votes": []})
        entry["votes"].append(int(pred))

    n_correct = 0
    for entry in by_video.values():
        vote_counts = np.bincount(entry["votes"], minlength=len(CLASSES))
        majority_pred = int(np.argmax(vote_counts))
        n_correct += int(majority_pred == entry["true_idx"])

    return {
        "n_test_videos": len(by_video),
        "per_video_accuracy_majority_vote": n_correct / len(by_video) if by_video else None,
    }


def train_classifier(seed: int = SEED, max_epochs: int = MAX_EPOCHS) -> dict:
    tf.keras.utils.set_random_seed(seed)

    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")

    y_train_onehot = utils.to_categorical(y_train, num_classes=len(CLASSES))
    y_val_onehot = utils.to_categorical(y_val, num_classes=len(CLASSES))

    class_names = _class_names_from_bronze_manifest()
    class_weights = compute_class_weights(class_names)  # {class_idx: peso}, do manifest bronze

    model = _build_and_compile_model()
    train_seq = _AugmentedTrainSequence(X_train, y_train_onehot, BATCH_SIZE, seed)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        train_seq,
        validation_data=(X_val, y_val_onehot),
        epochs=max_epochs,
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE
            ),
            callbacks.ModelCheckpoint(str(CHECKPOINT_PATH), monitor="val_loss", save_best_only=True),
        ],
        verbose=2,
    )

    evaluation = _evaluate(model, X_test, y_test)

    report = {
        "seed": seed,
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": history.history["loss"][-1],
        "final_train_accuracy": history.history["accuracy"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "final_val_accuracy": history.history["val_accuracy"][-1],
        "class_weights": {CLASSES[idx]: w for idx, w in class_weights.items()},
        "checkpoint_path": str(CHECKPOINT_PATH),
        **evaluation,
        "hard_floor_0_7511_passed": evaluation["test_accuracy_overall"] > HARD_FLOOR_ACCURACY,
        "stretch_reference_0_935_gap": STRETCH_REFERENCE_ACCURACY - evaluation["test_accuracy_overall"],
    }

    TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["training_report_path"] = str(TRAINING_REPORT_PATH)
    return report


if __name__ == "__main__":
    import sys
    cli_max_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_EPOCHS
    summary = train_classifier(max_epochs=cli_max_epochs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
