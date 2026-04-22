#!/usr/bin/env python3
"""
Visualize champion embeddings using t-SNE.

Extracts the ally and enemy embedding weights from a trained model,
reduces to 2D with t-SNE, and saves a scatter where each point is a champion.

Usage:
    python visualize_embeddings.py
    python visualize_embeddings.py --model-dir model_15.8_5000_matches
    python visualize_embeddings.py --model-dir model_15.8_5000_matches --output model_15.8_5000_matches/embedding_visualization.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE

from predict import Predictor

# ─── Config ───────────────────────────────────────────────────────────────────

PERPLEXITY  = 30
RANDOM_SEED = 42


# ─── t-SNE ────────────────────────────────────────────────────────────────────

def run_tsne(vecs: np.ndarray) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    ).fit_transform(vecs)


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_tsne(coords: np.ndarray, labels: list[str], title: str, ax: plt.Axes) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.7, linewidths=0)
    for i, name in enumerate(labels):
        txt = ax.text(
            coords[i, 0], coords[i, 1], name,
            fontsize=5.5, ha="center", va="bottom",
        )
        txt.set_path_effects([
            pe.Stroke(linewidth=1.5, foreground="white"),
            pe.Normal(),
        ])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize champion embeddings with t-SNE.")
    parser.add_argument(
        "--model-dir", default=None,
        help="Model directory to load (default: latest model_*_matches/ found)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path (default: embedding_visualization.png in CWD)",
    )
    args = parser.parse_args()

    predictor = Predictor(model_dir=args.model_dir)
    model = predictor.model
    vocab = predictor.vocab

    ally_weights  = model.get_layer("ally_emb").get_weights()[0]
    enemy_weights = model.get_layer("enemy_emb").get_weights()[0]

    names      = [vocab.decode(i) for i in range(1, vocab.size)]
    ally_vecs  = ally_weights[1:]
    enemy_vecs = enemy_weights[1:]

    print("Running t-SNE on ally embeddings…")
    ally_2d  = run_tsne(ally_vecs)
    print("Running t-SNE on enemy embeddings…")
    enemy_2d = run_tsne(enemy_vecs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Champion Embedding Space (t-SNE)", fontsize=15, fontweight="bold", y=1.01)

    plot_tsne(ally_2d,  names, "Ally Embeddings\n(synergy clusters)",      ax1)
    plot_tsne(enemy_2d, names, "Enemy Embeddings\n(counter-pick clusters)", ax2)

    plt.tight_layout()
    out = args.output or "embedding_visualization.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
