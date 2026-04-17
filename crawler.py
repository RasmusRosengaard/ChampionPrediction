#!/usr/bin/env python3
"""
League of Legends EUW ranked match data crawler.
Snowballs from seed summoners through participant PUUIDs to build a dataset.
"""

import json
import os
import sys
import time
import logging
from collections import deque
from pathlib import Path
from urllib.parse import quote
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

API_KEY = os.getenv("RIOT_API_KEY", "")
if not API_KEY:
    sys.exit("RIOT_API_KEY not set — add it to your .env file.")

SEED_SUMMONERS = [
    "r3r0ni#exe",
    "r3r0ni#exe1",
]

TARGET_MATCHES = 10000
OUTPUT_DIR = "match_files"

# ─── Constants ────────────────────────────────────────────────────────────────

PLATFORM_URL = "https://euw1.api.riotgames.com"
REGIONAL_URL = "https://europe.api.riotgames.com"

STATE_FILE = "crawler_state.json"
RANKS_FILE = "player_ranks.json"

# Rate limits (stay under dev key limits with a small safety margin)
RATE_LIMIT_SHORT = 18      # max 20/s — leave 2 buffer
RATE_LIMIT_LONG  = 95      # max 100/2min — leave 5 buffer
RATE_WINDOW_SHORT = 1.0
RATE_WINDOW_LONG  = 120.0

TIER_ORDER = {
    "IRON": 1, "BRONZE": 2, "SILVER": 3, "GOLD": 4,
    "PLATINUM": 5, "EMERALD": 6, "DIAMOND": 7,
    "MASTER": 8, "GRANDMASTER": 9, "CHALLENGER": 10,
}
TIER_NAMES = {v: k for k, v in TIER_ORDER.items()}

# ─── Logging ──────────────────────────────────────────────────────────────────

_file_handler = logging.FileHandler("crawler.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("\r%(message)s\033[K"))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
log = logging.getLogger(__name__)


def cprint(msg: str):
    """Print a full line, clearing any leftover progress bar characters."""
    print(f"\r{msg}\033[K")


# ─── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self):
        self._short: list[float] = []
        self._long: list[float] = []
        self.total_requests = 0
        self._start = time.monotonic()

    def wait(self):
        now = time.monotonic()

        # Evict stale timestamps
        self._short = [t for t in self._short if now - t < RATE_WINDOW_SHORT]
        self._long  = [t for t in self._long  if now - t < RATE_WINDOW_LONG]

        # Block on short window
        if len(self._short) >= RATE_LIMIT_SHORT:
            gap = RATE_WINDOW_SHORT - (now - self._short[0]) + 0.05
            if gap > 0:
                time.sleep(gap)
            now = time.monotonic()
            self._short = [t for t in self._short if now - t < RATE_WINDOW_SHORT]

        # Block on long window
        if len(self._long) >= RATE_LIMIT_LONG:
            gap = RATE_WINDOW_LONG - (now - self._long[0]) + 0.1
            if gap > 0:
                cprint(f"[rate] 2-min window full — sleeping {gap:.1f}s")
                log.info(f"Long window throttle: sleeping {gap:.1f}s")
                time.sleep(gap)
            now = time.monotonic()
            self._long = [t for t in self._long if now - t < RATE_WINDOW_LONG]

        now = time.monotonic()
        self._short.append(now)
        self._long.append(now)
        self.total_requests += 1

    @property
    def rate_per_minute(self) -> float:
        elapsed = time.monotonic() - self._start
        return (self.total_requests / elapsed * 60) if elapsed > 1 else 0.0


limiter = RateLimiter()


# ─── HTTP ─────────────────────────────────────────────────────────────────────

def riot_get(url: str, params: dict | None = None, retries: int = 5) -> dict | list | None:
    headers = {"X-Riot-Token": API_KEY}
    params = params or {}

    for attempt in range(retries):
        limiter.wait()
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        except requests.RequestException as exc:
            log.warning(f"Network error (attempt {attempt+1}): {url} — {exc}")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 404:
            log.debug(f"404 skipped: {url}")
            return None

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            cprint(f"[429] Rate limited — backing off {retry_after}s")
            log.warning(f"429 on {url} — backing off {retry_after}s")
            time.sleep(retry_after + 1)
            continue

        log.error(f"HTTP {resp.status_code} on {url}")
        time.sleep(2 * (attempt + 1))

    return None


# ─── API Helpers ──────────────────────────────────────────────────────────────

def get_puuid_by_riot_id(riot_id: str) -> str | None:
    """Resolves 'gameName#tagLine' to a PUUID via the Account API."""
    if "#" not in riot_id:
        cprint(f"[warn] Seed {riot_id!r} must be in 'gameName#tagLine' format")
        return None
    game_name, tag_line = riot_id.split("#", 1)
    url = f"{REGIONAL_URL}/riot/account/v1/accounts/by-riot-id/{quote(game_name)}/{quote(tag_line)}"
    account = riot_get(url)
    if not isinstance(account, dict):
        return None
    return account.get("puuid")


def get_match_ids(puuid: str, count: int = 100) -> list[str]:
    url = f"{REGIONAL_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    result = riot_get(url, params={"queue": 420, "type": "ranked", "count": count})
    return result if isinstance(result, list) else []


def get_match(match_id: str) -> dict | None:
    url = f"{REGIONAL_URL}/lol/match/v5/matches/{match_id}"
    return riot_get(url)  # type: ignore[return-value]


def get_rank(puuid: str) -> str | None:
    """Returns the RANKED_SOLO_5x5 tier string, e.g. 'GOLD', or None."""
    url = f"{PLATFORM_URL}/lol/league/v4/entries/by-puuid/{puuid}"
    entries = riot_get(url)
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if entry.get("queueType") == "RANKED_SOLO_5x5":
            return entry.get("tier")
    return None


# ─── Tier Utilities ───────────────────────────────────────────────────────────

def tier_to_num(tier: str) -> int:
    return TIER_ORDER.get(tier.upper(), 4)  # default: GOLD


def num_to_tier(n: float) -> str:
    return TIER_NAMES[max(1, min(10, round(n)))]


# ─── Persistence ──────────────────────────────────────────────────────────────

def load_state() -> tuple[set[str], set[str]]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("processed_puuids", [])), set(data.get("downloaded_matches", []))
    return set(), set()


def save_state(processed: set[str], downloaded: set[str]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"processed_puuids": list(processed), "downloaded_matches": list(downloaded)},
            f,
        )


def load_ranks() -> dict[str, str]:
    if os.path.exists(RANKS_FILE):
        with open(RANKS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ranks(ranks: dict[str, str]) -> None:
    with open(RANKS_FILE, "w", encoding="utf-8") as f:
        json.dump(ranks, f, indent=2)


# ─── Progress ─────────────────────────────────────────────────────────────────

def print_progress(matches: int, players: int, queue_size: int) -> None:
    rate = limiter.rate_per_minute
    sys.stdout.write(
        f"\r  matches {matches:>5}/{TARGET_MATCHES}"
        f"  |  players {players:>6}"
        f"  |  queue {queue_size:>5}"
        f"  |  requests {limiter.total_requests:>6}"
        f"  |  {rate:>5.1f} req/min  "
    )
    sys.stdout.flush()


# ─── Crawler ──────────────────────────────────────────────────────────────────

def resolve_tier(puuid: str, estimated_tier: str, ranks: dict[str, str]) -> str:
    """
    Return a confirmed rank for this PUUID, fetching from API if needed.
    Falls back to the inherited estimated_tier so we always have something.
    """
    if puuid in ranks:
        return ranks[puuid]

    fetched = get_rank(puuid)
    if fetched:
        ranks[puuid] = fetched
        return fetched

    # No rank data available — inherit parent's tier as a rough proxy
    ranks[puuid] = estimated_tier
    return estimated_tier


def crawl() -> None:
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    processed_puuids, state_matches = load_state()
    ranks = load_ranks()

    # Disk is the source of truth for what's actually downloaded
    downloaded_matches = {f.stem for f in Path(OUTPUT_DIR).glob("*.json")}

    # Detect manual reset: if disk has far fewer files than state expected,
    # the processed set is stale too — clear it so players get re-queued
    if len(downloaded_matches) < len(state_matches) * 0.5:
        cprint(
            f"[info] Reset detected (disk={len(downloaded_matches)} vs state={len(state_matches)})"
            f" — clearing processed player set"
        )
        processed_puuids.clear()

    # Queue items: (puuid, estimated_tier)
    queue: deque[tuple[str, str]] = deque()
    seen_puuids: set[str] = set(processed_puuids)

    # ── Bootstrap from seed summoners ────────────────────────────────────────
    cprint(f"[boot] Resolving {len(SEED_SUMMONERS)} seed summoner(s)…")
    for name in SEED_SUMMONERS:
        puuid = get_puuid_by_riot_id(name)
        if not puuid:
            cprint(f"[warn] Seed not found: {name!r}")
            log.warning(f"Seed summoner not found: {name}")
            continue
        if puuid in seen_puuids:
            continue
        tier = get_rank(puuid) or "GOLD"
        ranks[puuid] = tier
        queue.append((puuid, tier))
        seen_puuids.add(puuid)
        cprint(f"[seed] {name!r} → {tier}")
        log.info(f"Seeded {name!r} puuid={puuid[:12]}… tier={tier}")

    save_ranks(ranks)

    matches_collected = len(downloaded_matches)
    save_checkpoint_counter = 0

    cprint(
        f"[start] target={TARGET_MATCHES}  already_have={matches_collected}"
        f"  queue={len(queue)}"
    )
    log.info(
        f"Crawl started. target={TARGET_MATCHES} have={matches_collected} queue={len(queue)}"
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    while queue and matches_collected < TARGET_MATCHES:
        puuid, estimated_tier = queue.popleft()

        if puuid in processed_puuids:
            continue

        # Confirm (or fetch) this player's actual rank
        tier = resolve_tier(puuid, estimated_tier, ranks)

        # Fetch their ranked match list
        match_ids = get_match_ids(puuid)
        log.info(f"[player] {puuid[:16]}…  tier={tier}  match_ids={len(match_ids)}")

        for match_id in match_ids:
            if matches_collected >= TARGET_MATCHES:
                break

            out_path = Path(OUTPUT_DIR) / f"{match_id}.json"

            # Skip if already on disk (covers both state-tracked and orphaned files)
            if match_id in downloaded_matches or out_path.exists():
                downloaded_matches.add(match_id)
                continue

            log.info(f"  [fetch]  #{matches_collected + 1:<5}  {match_id}")
            match_data = get_match(match_id)
            if not match_data:
                downloaded_matches.add(match_id)
                continue

            info = match_data.get("info", {})
            champs = [p.get("championName", "?") for p in info.get("participants", [])]
            duration_s = info.get("gameDuration", 0)
            duration = f"{duration_s // 60}m{duration_s % 60:02d}s"

            game_version = info.get("gameVersion", "")
            patch = ".".join(game_version.split(".")[:2])  # "14.10.123.456" → "14.10"

            wrapped = {
                "metadata": {
                    "crawler_tier": tier,
                    "patch": patch,
                },
                "match": match_data,
            }
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(wrapped, fh)

            downloaded_matches.add(match_id)
            matches_collected += 1
            log.info(
                f"  [saved]  #{matches_collected:<5}  {match_id}"
                f"  {duration}  [{', '.join(champs)}]"
            )

            # Snowball: enqueue every participant we haven't seen yet
            participants: list[str] = match_data.get("metadata", {}).get("participants", [])
            for p_puuid in participants:
                if p_puuid not in seen_puuids:
                    seen_puuids.add(p_puuid)
                    # Inherit current player's tier as initial estimate
                    queue.append((p_puuid, tier))

        processed_puuids.add(puuid)
        save_checkpoint_counter += 1

        # Checkpoint every 10 players processed
        if save_checkpoint_counter >= 10:
            save_state(processed_puuids, downloaded_matches)
            save_ranks(ranks)
            save_checkpoint_counter = 0

        print_progress(matches_collected, len(seen_puuids), len(queue))

    # ── Finish ────────────────────────────────────────────────────────────────
    print()  # newline after progress bar
    save_state(processed_puuids, downloaded_matches)
    save_ranks(ranks)

    cprint(
        f"[done] collected={matches_collected}  players_processed={len(processed_puuids)}"
        f"  total_requests={limiter.total_requests}"
    )
    log.info(
        f"Crawl complete. matches={matches_collected} players={len(processed_puuids)}"
        f" requests={limiter.total_requests}"
    )


if __name__ == "__main__":
    crawl()
