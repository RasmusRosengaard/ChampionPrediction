#!/usr/bin/env python3
"""
Champion pick predictor — inference script.

Given your current ally picks, enemy picks, and the role you need to fill,
returns the top-N recommended champions.

Usage:
    python predict.py
    Edit the SCENARIOS list at the bottom to try different compositions.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import (
    ALLY_SLOTS,
    ENEMY_SLOTS,
    MODEL_DIR,
    PAD_IDX,
    ROLE_TO_IDX,
    ROLES,
    ChampionVocab,
    DraftModel,
    EMBEDDING_DIM,
    ROLE_EMBED_DIM,
    HIDDEN_UNITS,
    DROPOUT_RATE,
)

# ─── Config ───────────────────────────────────────────────────────────────────

TOP_N = 10   # how many recommendations to show per query


def find_latest_model_dir() -> str:
    """Returns the model folder with the highest match count, e.g. model_5000_matches."""
    candidates = sorted(
        Path(".").glob(f"{MODEL_DIR}_*_matches"),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    # Fall back to plain MODEL_DIR for backwards compatibility
    return MODEL_DIR


# ─── Loader ───────────────────────────────────────────────────────────────────

class Predictor:
    """Loads a saved model + vocab and exposes a recommend() method."""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = find_latest_model_dir()
            print(f"Using model: {model_dir}")
        vocab_path  = os.path.join(model_dir, "vocab.pkl")
        model_path  = os.path.join(model_dir, "draft_model.keras")
        weights_path = os.path.join(model_dir, "best.weights.h5")

        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Vocab not found at {vocab_path} — run train.py first.")
        if not Path(model_path).exists() and not Path(weights_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path} — run train.py first.")

        self.vocab = ChampionVocab.load(vocab_path)

        self.model = DraftModel(
            vocab_size=self.vocab.size,
            embedding_dim=EMBEDDING_DIM,
            role_embed_dim=ROLE_EMBED_DIM,
            hidden_units=HIDDEN_UNITS,
            dropout_rate=DROPOUT_RATE,
        )

        # Warm-up pass to build weights before loading
        dummy = {
            "ally":  tf.zeros((1, ALLY_SLOTS),  dtype=tf.int32),
            "enemy": tf.zeros((1, ENEMY_SLOTS), dtype=tf.int32),
            "role":  tf.zeros((1,),             dtype=tf.int32),
        }
        self.model(dummy, training=False)

        if Path(model_path).exists():
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={"DraftModel": DraftModel},
            )
        else:
            self.model.load_weights(weights_path)

        print(f"Loaded model — {self.vocab}")

    def recommend(
        self,
        ally_picks: list[str],
        enemy_picks: list[str],
        role: str,
        top_n: int = TOP_N,
        exclude_picked: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Returns a ranked list of (champion_name, score) tuples.

        ally_picks:   up to 4 champion names already on your team
        enemy_picks:  up to 5 champion names on the enemy team
        role:         one of TOP / JUNGLE / MID / BOTTOM / UTILITY
        exclude_picked: remove already-picked champs from results
        """
        role = role.upper()
        role = "UTILITY" if role == "SUPPORT" else role
        if role not in ROLE_TO_IDX:
            raise ValueError(f"Unknown role {role!r}. Valid: {ROLES + ['SUPPORT']}")

        def encode_team(picks: list[str], slots: int) -> np.ndarray:
            ids = [self.vocab.encode(c) for c in picks]
            ids = (ids + [PAD_IDX] * slots)[:slots]
            return np.array([ids], dtype=np.int32)

        inputs = {
            "ally":  encode_team(ally_picks,  ALLY_SLOTS),
            "enemy": encode_team(enemy_picks, ENEMY_SLOTS),
            "role":  np.array([[ROLE_TO_IDX[role]]], dtype=np.int32).reshape(1),
        }

        logits = self.model(inputs, training=False)[0].numpy()  # [vocab_size]
        probs  = _softmax(logits)

        already_picked = set(ally_picks + enemy_picks) if exclude_picked else set()

        ranked = [
            (self.vocab.decode(idx), float(probs[idx]))
            for idx in np.argsort(probs)[::-1]
            if idx != PAD_IDX and self.vocab.decode(idx) not in already_picked
        ]
        return ranked[:top_n]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def print_recommendations(
    role: str,
    ally_picks: list[str],
    enemy_picks: list[str],
    results: list[tuple[str, float]],
) -> None:
    print("\n" + "─" * 50)
    print(f"  Role      : {role}")
    print(f"  Allies    : {', '.join(ally_picks) if ally_picks else '(none)'}")
    print(f"  Enemies   : {', '.join(enemy_picks) if enemy_picks else '(none)'}")
    print(f"  Top {len(results)} picks:")
    for rank, (champ, score) in enumerate(results, 1):
        bar = "█" * int(score * 200)
        print(f"    {rank:>2}. {champ:<20} {score:.4f}  {bar}")
    print("─" * 50)


# ─── Example Scenarios ────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "role": "MID",
        "ally_picks":  ["Jinx", "Thresh", "Garen", "Vi"],
        "enemy_picks": ["Zed", "Yasuo", "Caitlyn", "Leona", "Malphite"],
    },
    {
        "role": "SUPPORT",
        "ally_picks":  ["Jinx", "Orianna", "Graves", "Darius"],
        "enemy_picks": ["Draven", "Pyke", "Katarina", "Hecarim", "Camille"],
    },
    {
        "role": "TOP",
        "ally_picks":  ["Ahri", "Lee Sin", "Jhin", "Nautilus"],
        "enemy_picks": ["Darius", "Zed", "Ezreal", "Lulu", "Graves"],
    },
    {
        "role": "JUNGLE",
        "ally_picks":  ["Viktor", "Garen", "Jinx", "Blitzcrank"],
        "enemy_picks": ["Riven", "Syndra", "Caitlyn", "Thresh", "Shyvana"],
    },
    {
        "role": "BOTTOM",
        "ally_picks":  ["Malphite", "Elise", "Zoe", "Lulu"],
        "enemy_picks": ["Jinx", "Thresh", "Yasuo", "Diana", "Fiora"],
    },
    {
        "role": "JUNGLE",
        "ally_picks":  ["Varus", "Ryze", "Senna", "Leona"],
        "enemy_picks": ["Sion", "Nafari", "Lux", "Sivir", "Bard"],
    },
]


def main() -> None:
    predictor = Predictor()

    for scenario in SCENARIOS:
        results = predictor.recommend(
            ally_picks=scenario["ally_picks"],
            enemy_picks=scenario["enemy_picks"],
            role=scenario["role"],
            top_n=TOP_N,
        )
        print_recommendations(
            role=scenario["role"],
            ally_picks=scenario["ally_picks"],
            enemy_picks=scenario["enemy_picks"],
            results=results,
        )


if __name__ == "__main__":
    main()
