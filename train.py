#!/usr/bin/env python3
"""
Champion pick predictor — trains a model on crawled match data.

Given 4 ally picks, 5 enemy picks, and a target role, predicts the
optimal champion to pick. Learns synergy and counter-pick signals
simultaneously from the final compositions in each match.
"""

import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────

MATCH_DIR        = "match_files"
MODEL_DIR        = "model"

# Each champion is represented as a list of 64 learned numbers (its "stats sheet").
# Champions with similar roles/playstyles end up with similar values.
# Ally and enemy embeddings are SEPARATE tables — what a champion means as a
# teammate is learned independently from what it means as a threat to play against.
# Larger = more expressive but slower to train.
EMBEDDING_DIM    = 64

# Same idea as EMBEDDING_DIM but for the 5 roles (TOP/JG/MID/BOT/SUPPORT).
# Smaller because there are only 5 roles vs ~170 champions.
ROLE_EMBED_DIM   = 16

# Two fully-connected layers between input and output.
# [256] learns low-level patterns ("these two champs appear together often").
# [128] learns higher-level patterns ("this combo tends to win in this role").
# More units = more capacity to learn, but risks memorising the training data.
HIDDEN_UNITS     = [256, 128]

# During training, 30% of neurons are randomly switched off each pass.
# Forces the model to not rely on any single neuron — prevents overfitting
# (memorising training data instead of learning generalizable patterns).
# Dropout is disabled automatically during prediction.
DROPOUT_RATE     = 0.3

# How many samples are processed in one forward/backward pass.
# Larger = faster training, but needs more RAM.
BATCH_SIZE       = 512

# Maximum number of full passes through the training data.
# EarlyStopping (patience=5) will stop before this if validation loss plateaus.
EPOCHS           = 1000

LEARNING_RATE    = 1e-3

# 90% of samples train the model, 10% are held out as a validation set.
# The model never learns from validation data — it measures generalisation.
VALIDATION_SPLIT = 0.1

# Only train on a specific patch, e.g. "14.10". None = all patches.
# Retrain with the current patch after each major balance update.
PATCH_FILTER     = None

ROLES            = ["TOP", "JUNGLE", "MID", "BOTTOM", "UTILITY"]
ROLE_TO_IDX      = {r: i for i, r in enumerate(ROLES)}
NUM_ROLES        = len(ROLES)
ALLY_SLOTS       = 4      # ally team minus the target pick
ENEMY_SLOTS      = 5
PAD_IDX          = 0      # index 0 reserved for padding / unknown champion

# ─── Vocabulary ───────────────────────────────────────────────────────────────

class ChampionVocab:
    """
    Bidirectional mapping between champion name strings and integer indices.
    Index 0 is reserved as PAD / UNKNOWN and never assigned to a real champion.
    """

    def __init__(self):
        self._name_to_idx: dict[str, int] = {}
        self._idx_to_name: list[str] = ["<PAD>"]

    def build(self, names: list[str]) -> None:
        for name in sorted(set(n for n in names if n)):
            if name not in self._name_to_idx:
                self._name_to_idx[name] = len(self._idx_to_name)
                self._idx_to_name.append(name)

    def encode(self, name: str) -> int:
        return self._name_to_idx.get(name, PAD_IDX)

    def decode(self, idx: int) -> str:
        return self._idx_to_name[idx] if 0 <= idx < len(self._idx_to_name) else "<UNKNOWN>"

    @property
    def size(self) -> int:
        return len(self._idx_to_name)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"name_to_idx": self._name_to_idx, "idx_to_name": self._idx_to_name}, f)

    @classmethod
    def load(cls, path: str) -> "ChampionVocab":
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls()
        vocab._name_to_idx = data["name_to_idx"]
        vocab._idx_to_name = data["idx_to_name"]
        return vocab

    def __repr__(self) -> str:
        return f"ChampionVocab({self.size - 1} champions)"


# ─── Data Parsing ─────────────────────────────────────────────────────────────

class Sample:
    """One training example: predict target_champ given team context and role."""
    __slots__ = ("ally_champs", "enemy_champs", "role_idx", "target_champ", "weight")

    def __init__(
        self,
        ally_champs: list[str],
        enemy_champs: list[str],
        role_idx: int,
        target_champ: str,
        weight: float = 1.0,
    ):
        self.ally_champs = ally_champs
        self.enemy_champs = enemy_champs
        self.role_idx = role_idx
        self.target_champ = target_champ
        self.weight = weight


class MatchParser:
    """Extracts Sample objects from a single Riot match JSON blob."""

    _VALID_ROLES = set(ROLES)

    @classmethod
    def parse(cls, match_json: dict) -> list[Sample]:
        """
        Generates one sample per player on the WINNING team only.
        The label is the champion that was picked in that role on the winning side,
        teaching the model what champion SHOULD be picked to maximise win chance —
        not just what people tend to pick.
        """
        info = match_json.get("match", {}).get("info", {})
        participants = info.get("participants", [])
        if len(participants) != 10:
            return []

        teams: dict[int, list[dict]] = defaultdict(list)
        for p in participants:
            teams[p.get("teamId", 0)].append(p)

        samples = []
        for p in participants:
            # Skip losers — we only learn from winning compositions
            if not p.get("win", False):
                continue

            role = p.get("teamPosition") or p.get("individualPosition", "")
            if role not in cls._VALID_ROLES:
                continue
            target = p.get("championName", "")
            if not target:
                continue

            team_id = p.get("teamId")
            allies = [
                a.get("championName", "")
                for a in teams[team_id]
                if a is not p and a.get("championName")
            ]
            enemies = [
                e.get("championName", "")
                for e in participants
                if e.get("teamId") != team_id and e.get("championName")
            ]

            samples.append(Sample(allies, enemies, ROLE_TO_IDX[role], target, weight=1.0))

        return samples


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DraftDataset:
    """
    Loads all match files from disk, builds a champion vocabulary, and
    produces train/validation tf.data.Dataset pairs.
    """

    def __init__(self, match_dir: str):
        self.match_dir = Path(match_dir)
        self.vocab = ChampionVocab()
        self._samples: list[Sample] = []

    def load(self, patch_filter: str | None = PATCH_FILTER) -> int:
        files = list(self.match_dir.glob("*.json"))
        print(f"Loading {len(files)} match files…")
        if patch_filter:
            print(f"Patch filter active: only using matches from patch {patch_filter}")

        all_names: list[str] = []
        raw: list[Sample] = []
        skipped_patch = 0

        for i, fpath in enumerate(files):
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if patch_filter and data.get("metadata", {}).get("patch") != patch_filter:
                skipped_patch += 1
                continue

            for s in MatchParser.parse(data):
                all_names.extend(s.ally_champs)
                all_names.extend(s.enemy_champs)
                all_names.append(s.target_champ)
                raw.append(s)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(files)} files — {len(raw)} samples", end="\r")

        if skipped_patch:
            print(f"\nSkipped {skipped_patch} matches not on patch {patch_filter}.")
        print(f"\nLoaded {len(raw)} samples. Building vocabulary…")
        self.vocab.build(all_names)
        self._samples = raw
        print(f"{self.vocab}")
        return len(raw)

    def _encode(self, s: Sample) -> tuple[np.ndarray, np.ndarray, int, int, float]:
        ally = [self.vocab.encode(c) for c in s.ally_champs]
        ally = (ally + [PAD_IDX] * ALLY_SLOTS)[:ALLY_SLOTS]

        enemy = [self.vocab.encode(c) for c in s.enemy_champs]
        enemy = (enemy + [PAD_IDX] * ENEMY_SLOTS)[:ENEMY_SLOTS]

        return (
            np.array(ally,  dtype=np.int32),
            np.array(enemy, dtype=np.int32),
            np.int32(s.role_idx),
            np.int32(self.vocab.encode(s.target_champ)),
            np.float32(s.weight),
        )

    def to_tf_datasets(
        self, shuffle: bool = True
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        encoded = [self._encode(s) for s in self._samples]

        # Shuffle before splitting so val set covers all champions/roles
        if shuffle:
            random.shuffle(encoded)

        n = len(encoded)
        val_n = max(1, int(n * VALIDATION_SPLIT))
        val_enc, train_enc = encoded[:val_n], encoded[val_n:]

        def make_ds(items: list, reshuffle: bool) -> tf.data.Dataset:
            allies  = np.stack([e[0] for e in items])
            enemies = np.stack([e[1] for e in items])
            roles   = np.array([e[2] for e in items], dtype=np.int32)
            targets = np.array([e[3] for e in items], dtype=np.int32)
            weights = np.array([e[4] for e in items], dtype=np.float32)
            ds = tf.data.Dataset.from_tensor_slices((
                {"ally": allies, "enemy": enemies, "role": roles},
                targets,
                weights,
            ))
            if reshuffle:
                ds = ds.shuffle(buffer_size=min(len(items), 50_000), reshuffle_each_iteration=True)
            return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return make_ds(train_enc, reshuffle=True), make_ds(val_enc, reshuffle=False)


# ─── Model ────────────────────────────────────────────────────────────────────

class MaskedMeanPool(tf.keras.layers.Layer):
    """
    Averages champion embeddings across a team, ignoring PAD (index 0) slots.
    """

    def call(self, embeddings: tf.Tensor, ids: tf.Tensor) -> tf.Tensor:
        mask   = tf.cast(tf.not_equal(ids, PAD_IDX), tf.float32)  # [B, slots]
        mask   = tf.expand_dims(mask, -1)                          # [B, slots, 1]
        pooled = tf.reduce_sum(embeddings * mask, axis=1)          # [B, emb_dim]
        count  = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)      # [B, 1]
        return pooled / count                                       # [B, emb_dim]

    def get_config(self) -> dict:
        return super().get_config()


class DraftModel(tf.keras.Model):
    """
    Predicts champion pick from team context.

    Input dict keys:
      ally  [B, 4]  — champion IDs of the 4 other allies  (0 = PAD)
      enemy [B, 5]  — champion IDs of the 5 enemies       (0 = PAD)
      role  [B]     — role index 0–4

    Output:
      logits [B, vocab_size] — unnormalized score for each champion
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        role_embed_dim: int,
        hidden_units: list[int],
        dropout_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.role_embed_dim = role_embed_dim
        self.hidden_units  = hidden_units
        self.dropout_rate  = dropout_rate

        # Separate embeddings: what a champion means as an ally vs. as an enemy
        self.ally_emb  = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="ally_emb")
        self.enemy_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="enemy_emb")
        self.role_emb  = tf.keras.layers.Embedding(NUM_ROLES, role_embed_dim, name="role_emb")
        self.pool      = MaskedMeanPool(name="masked_mean_pool")

        self.hidden = [
            tf.keras.layers.Dense(u, activation="relu", name=f"dense_{i}")
            for i, u in enumerate(hidden_units)
        ]
        self.drops = [
            tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i}")
            for i in range(len(hidden_units))
        ]
        self.out = tf.keras.layers.Dense(vocab_size, name="logits")

    def call(self, inputs: dict, training: bool = False) -> tf.Tensor:
        ally_ids  = inputs["ally"]
        enemy_ids = inputs["enemy"]
        role_ids  = inputs["role"]

        ally_vec  = self.pool(self.ally_emb(ally_ids),   ally_ids)
        enemy_vec = self.pool(self.enemy_emb(enemy_ids), enemy_ids)
        role_vec  = self.role_emb(role_ids)

        x = tf.concat([ally_vec, enemy_vec, role_vec], axis=-1)
        for layer, drop in zip(self.hidden, self.drops):
            x = drop(layer(x), training=training)
        return self.out(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "vocab_size":    self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "role_embed_dim": self.role_embed_dim,
            "hidden_units":  self.hidden_units,
            "dropout_rate":  self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "DraftModel":
        return cls(**config)


# ─── Training ─────────────────────────────────────────────────────────────────

def build_and_compile(vocab_size: int) -> DraftModel:
    model = DraftModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        role_embed_dim=ROLE_EMBED_DIM,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT_RATE,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def main() -> None:
    Path(MODEL_DIR).mkdir(exist_ok=True)

    dataset = DraftDataset(MATCH_DIR)
    if dataset.load() == 0:
        print("No samples found — run crawler.py first.")
        return

    dataset.vocab.save(os.path.join(MODEL_DIR, "vocab.pkl"))
    train_ds, val_ds = dataset.to_tf_datasets()

    model = build_and_compile(dataset.vocab.size)

    # Warm-up pass to materialise weights before summary
    for batch in train_ds.take(1):
        model(batch[0], training=False)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best.weights.h5"),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor="val_loss",
            verbose=1,
        ),
    ]

    print(f"\nTraining on {dataset.vocab.size - 1} champions. Target: {EPOCHS} epochs.\n")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    model.save(os.path.join(MODEL_DIR, "draft_model.keras"))
    print(f"\nSaved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
