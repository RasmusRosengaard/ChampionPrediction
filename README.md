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
# Get a free dev key at https://developer.riotgames.com

# 5. Collect match data
python crawler.py

# 6. Train the model
python train.py

# 7. Run predictions
python predict.py
```

---

## How It Works

The pipeline has three stages: **data collection**, **training**, and **prediction**.

### Stage 1 — Data Collection (`crawler.py`)

Crawls the Riot API starting from a list of seed summoners and snowballs through their games to build a dataset of ranked Solo/Duo matches.

```
Seed summoner → fetch their 100 matches → save each match as JSON
                                         → extract 10 player PUUIDs per match
                                         → add unseen PUUIDs to queue
                                         → repeat
```

Each match is saved as `match_files/{matchId}.json` and includes:
- Full participant data (champion, role, kills, items, etc.)
- Win/loss per player
- Game patch version
- Crawler metadata (elo tier of the discovering player)

The crawler respects Riot's dev key rate limits (20 req/s, 100 req/2min), backs off automatically on 429s, and can be interrupted and resumed at any time via `crawler_state.json`.

**Config** (top of `crawler.py`):
| Variable | Description |
|---|---|
| `SEED_SUMMONERS` | Starting players in `gameName#tagLine` format |
| `TARGET_MATCHES` | How many matches to collect before stopping |
| `OUTPUT_DIR` | Where match JSON files are saved |

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

**Config** (top of `train.py`):
| Variable | Description |
|---|---|
| `EMBEDDING_DIM` | Size of each champion's learned vector (default 64) |
| `HIDDEN_UNITS` | Neurons per layer, e.g. `[256, 128]` |
| `DROPOUT_RATE` | Fraction of neurons dropped during training to prevent overfitting |
| `EPOCHS` | Max training passes — early stopping will halt sooner if loss plateaus |
| `PATCH_FILTER` | Train on a specific patch only, e.g. `"14.10"`. `None` = all patches (recommended until you have 50k+ matches per patch) |

The trained model is saved to `model/draft_model.keras` and `model/vocab.pkl`.

---

### Stage 3 — Prediction (`predict.py`)

Loads the saved model and returns ranked champion recommendations for a given draft state. Works with partial picks — empty slots are masked out and ignored.

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

## Data & Model Quality

| Matches collected | Expected quality |
|---|---|
| 1,000–5,000 | Basic role identity, popular champions only |
| 10,000–50,000 | Decent synergy and counter-pick patterns |
| 100,000+ | Reliable recommendations across all champions |

More data = better predictions. Retrain after each major patch by setting `PATCH_FILTER` to the current patch number.

---

## Project Structure

```
ChampionPrediction/
├── crawler.py          # Riot API data crawler
├── train.py            # Model training
├── predict.py          # Champion recommendations
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── match_files/        # Saved match JSON (git ignored)
└── model/              # Trained model + vocab (git ignored)
```
