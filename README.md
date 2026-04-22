# ChampionPrediction

A League of Legends champion pick predictor trained on real EUW ranked match data.
Given your current ally picks, enemy picks, and the role you need to fill, the model recommends the best champions to pick — learning both synergy and counter-pick signals from thousands of games.

---

## Quick Start

**Requirements:** Python 3.11 (TensorFlow does not support 3.12+ yet)

```bash
# 1. Clone the repo
git clone https://github.com/RasmusRosengaard/ChampionPrediction.git
cd ChampionPrediction

# 2. Create and activate a virtual environment
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Riot API key
cp .env.example .env
# Open .env and replace RGAPI-your-key-here with your key
# Get a free dev key at https://developer.riotgames.com (expires every 24 hours)

# 5. Collect match data — training triggers automatically every 5,000 matches
python crawler.py

# 6. (Optional) Train manually on a specific patch
python train.py --patch 16.8

# 7. (Optional) Evaluate all trained models
python evaluate.py

# 8. Run predictions
python predict.py
```

---

## How It Works

The pipeline has four stages: **data collection**, **training**, **evaluation**, and **prediction**.

---

### Stage 1 — Data Collection (`crawler.py`)

Crawls the Riot API starting from a list of seed summoners and snowballs through their games to build a dataset of ranked Solo/Duo matches.

```
Seed summoner → fetch their N most-recent matches → check patch
                                                   → save if current patch
                                                   → extract 10 player PUUIDs per match
                                                   → add unseen PUUIDs to queue
                                                   → repeat
```

**Patch filtering** — At startup the crawler fetches the current patch from Riot's Data Dragon API (e.g. `16.8`). Only matches on the current patch are saved to disk. Matches from older patches are still used for snowballing (to discover more players) but are not counted or saved.

**Auto-training** — Every 5,000 collected matches the crawler automatically pauses, trains a new model, runs evaluation, and generates an embedding visualisation — all saved inside the new model folder — then resumes crawling.

Each match is saved as `match_files/{matchId}.json` and includes full participant data (champion, role, kills, items), win/loss per player, game patch version, and the elo tier of the discovering player.

The crawler respects Riot's dev key rate limits (20 req/s, 100 req/2min), backs off automatically on 429s, and can be interrupted and resumed at any time via `crawler_state.json`.

> **Note:** Riot development API keys expire every 24 hours. Regenerate your key at [developer.riotgames.com](https://developer.riotgames.com) if you get HTTP 401 errors.

**Config** (top of `crawler.py`):
| Variable | Description |
|---|---|
| `SEED_SUMMONERS` | Starting players in `gameName#tagLine` format |
| `TARGET_MATCHES` | How many matches to collect before stopping |
| `OUTPUT_DIR` | Where match JSON files are saved |
| `MATCHES_PER_PLAYER` | How many recent matches to fetch per player (default 10) |
| `TRAIN_EVERY_N_MATCHES` | Auto-train interval (default 5,000) |

To pin a specific patch without editing code, set the `PATCH_FILTER` environment variable:
```bash
PATCH_FILTER=16.8 python crawler.py
```

---

### Stage 2 — Training (`train.py`)

Trains a neural network on the collected matches. For every match, 5 training samples are generated — one per player on the **winning team only**:

> *"Given these 4 allies + 5 enemies + this role → what champion should be picked to win?"*

By training exclusively on winning compositions the model learns what champion **should** be picked to maximise win chance, not just what people tend to pick. Losing team picks are completely ignored.

**Model architecture:**
```
ally  [4 champion IDs] → Embedding (64d) → Masked Mean Pool → [64]  ─┐
enemy [5 champion IDs] → Embedding (64d) → Masked Mean Pool → [64]  ─┼→ Dense(256) → Dense(128) → scores[~170]
role  [1 role index]   → Embedding (16d)                    → [16]  ─┘
```

Ally and enemy embeddings are **separate** — what a champion means as a teammate is learned independently from what it means as a threat to play against.

Each version of the model is saved to its own folder named `model_{patch}_{n}_matches/` (e.g. `model_16.8_5000_matches/`), so you can compare across patches and data sizes. The folder also contains `evaluation_curves.png` and `embedding_visualization.png` generated automatically after each auto-training run.

**Config** (top of `train.py`):
| Variable | Description |
|---|---|
| `EMBEDDING_DIM` | Size of each champion's learned vector (default 64) |
| `HIDDEN_UNITS` | Neurons per layer, e.g. `[256, 128]` |
| `DROPOUT_RATE` | Fraction of neurons dropped during training to prevent overfitting |
| `EPOCHS` | Max training passes — early stopping halts sooner if loss plateaus |
| `PATCH_FILTER` | Default patch filter; overridden by `--patch` CLI arg |

```bash
python train.py                 # train on all saved matches
python train.py --patch 16.8    # train on patch 16.8 only
```

---

### Stage 3 — Evaluation (`evaluate.py`)

Evaluates trained model checkpoints on the same held-out test split and produces a comparison chart and results table.

```bash
python evaluate.py
# compare specific checkpoints:
python evaluate.py --model-dirs model_16.8_5000_matches model_16.8_10000_matches
# save results into a specific folder:
python evaluate.py --model-dirs model_16.8_5000_matches --output-dir model_16.8_5000_matches
```

**Metrics:**
| Metric | Description |
|---|---|
| **Loss** | Cross-entropy loss — lower is better |
| **Top-1 Accuracy** | Correct champion is the #1 pick |
| **Top-3 Accuracy** | Correct champion is in the top 3 picks |
| **Top-5 Accuracy** | Correct champion is in the top 5 picks |

Results are saved as `evaluation_curves.png` and `evaluation_results.json`. When triggered automatically by the crawler these are saved inside the model folder.

---

### Stage 4 — Prediction (`predict.py`)

Loads the latest saved model automatically and returns ranked champion recommendations for a given draft state. Works with partial picks — empty slots are masked out and ignored.

Edit the `SCENARIOS` list in `predict.py` to try different compositions:

```python
SCENARIOS = [
    {
        "role": "MID",
        "ally_picks":  ["Jinx", "Thresh", "Garen"],        # partial — only 3 allies known
        "enemy_picks": ["Zed", "Yasuo", "Caitlyn", "Leona", "Malphite"],
    },
]
```

Valid roles: `TOP`, `JUNGLE`, `MID`, `BOTTOM`, `SUPPORT` (or `UTILITY`)

**Example output:**
```
──────────────────────────────────────────────────
  Role      : TOP
  Allies    : Ahri, Lee Sin, Jhin, Nautilus
  Enemies   : Darius, Zed, Ezreal, Lulu, Graves
  Top 10 picks:
     1. Garen                0.0713  ██████████████
     2. Shen                 0.0483  █████████
     3. Ambessa              0.0469  █████████
     4. Aatrox               0.0394  ███████
     5. Malphite             0.0242  ████
──────────────────────────────────────────────────
```

---

### Embedding Visualisation (`visualize_embeddings.py`)

Runs t-SNE on the champion embedding weights and saves a scatter plot where similar champions cluster together — ally embeddings reflect synergy, enemy embeddings reflect counter-pick relationships.

```bash
python visualize_embeddings.py                                          # uses latest model
python visualize_embeddings.py --model-dir model_16.8_5000_matches      # specific model
python visualize_embeddings.py --model-dir model_16.8_5000_matches \
    --output model_16.8_5000_matches/embedding_visualization.png        # custom output path
```

This is run automatically after each auto-training and saved inside the model folder.

---

## Important: What the Model Optimises For

The model does **not** suggest lane counters. It has no concept of individual matchups — it sees the **entire allied team and the entire enemy team** as a whole, and recommends whatever champion winning teams most often picked in that situation.

In practice this means:
- It will often suggest lane counters, because good players do tend to pick them — but only when they also fit the overall composition.
- It may suggest a champion that *loses* the lane matchup but provides crucial team-wide value (engage, peel, split-push threat, etc.) that the enemy team struggles to deal with collectively.
- It will down-rank champions that are strong in isolation but clash with your existing ally picks (e.g. two champions that compete for the same win condition).

Think of it as *"what did winning teams pick in this exact draft situation?"* rather than *"what beats the enemy laner?"*

---

## Data & Model Quality

| Matches collected | Expected quality |
|---|---|
| 1,000–5,000 | Basic role identity, popular champions only |
| 10,000–50,000 | Decent synergy and counter-pick patterns |
| 100,000+ | Reliable recommendations across all champions |

More data = better predictions. The crawler automatically filters to the current patch, so models stay relevant after balance updates without any manual intervention.

---

## Project Structure

```
ChampionPrediction/
├── crawler.py                   # Riot API data crawler (patch-filtered, auto-trains)
├── train.py                     # Model training
├── evaluate.py                  # Model evaluation & comparison charts
├── predict.py                   # Champion recommendations
├── visualize_embeddings.py      # t-SNE embedding visualisation
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template
├── match_files/                 # Saved match JSON (git ignored)
└── model_{patch}_{n}_matches/   # Trained model checkpoints (git ignored)
    ├── draft_model.keras
    ├── best.weights.h5
    ├── vocab.pkl
    ├── history.json
    ├── evaluation_curves.png
    ├── evaluation_results.json
    └── embedding_visualization.png
```
