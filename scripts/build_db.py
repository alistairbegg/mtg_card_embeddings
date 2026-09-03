#!/usr/bin/env python3
"""Build a normalized DuckDB from public 17Lands draft/game/replay data.

For each dataset the builder scans each large source only once for normalization:
it establishes first-seen draft order from game_data, imports remaining draft picks in
one draft_data pass, imports remaining games/builds in one game_data pass, then imports
replay-derived hands, turns, states, zones and events in one replay_data pass.

The existing next_draft_index checkpoint remains authoritative, so databases created by
the previous draft-batched builder can resume without rebuilding completed drafts.
Additional source-phase checkpoints make the single-pass scans resumable after failure.
Wide draft rows use a small quote-aware metadata-prefix parser followed by NumPy's
compiled fromstring parser for the ~1,090 numeric pack/pool fields, avoiding per-card
Python parsing; other wide CSV rows use a C-level split fast path whenever quoting is
absent. Replay turn IDs are hashed once per active turn and reused across child tables.

The database preserves source truth rather than reconstructing ambiguous chronology.
The N in user_turn_N / oppo_turn_N is stored as source_turn_index. Every event keeps
the source turn bucket that contained it, while actual_turn_id is NULL when the
public replay export does not determine the active turn safely. Candidate mulligan
hands are stored as the observed seven-card candidates; the post-bottom starting
hand is not inferred.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

LOG = logging.getLogger("17lands-db")


class ProgressDisplay:
    """Low-overhead in-place progress display with a background refresh thread.

    The worker only updates a few counters. Rendering happens once per second on a
    daemon thread, so elapsed time continues moving while DuckDB is busy committing
    a large chunk. No third-party progress dependency is required.
    """

    def __init__(self, dataset_total: int, enabled: bool = True) -> None:
        self.dataset_total = dataset_total
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dataset_index = 0
        self._dataset_name = ""
        self._stage_name = ""
        self._stage_start = 0.0
        self._stage_start_rows = 0
        self._rows = 0
        self._total_rows: int | None = None
        self._fraction: float | None = None
        self._relevant: int | None = None
        self._relevant_label = "relevant"
        self._activity = ""
        self._last_line_len = 0
        self._thread: threading.Thread | None = None
        if self.enabled:
            self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._thread.start()

    def begin_dataset(self, index: int, name: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._dataset_index = index
            self._dataset_name = name

    def start_stage(
        self,
        name: str,
        *,
        start_rows: int = 0,
        total_rows: int | None = None,
        relevant_label: str = "relevant",
        activity: str = "scanning",
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._stage_name = name
            self._stage_start = time.monotonic()
            self._stage_start_rows = start_rows
            self._rows = start_rows
            self._total_rows = total_rows
            self._fraction = (start_rows / total_rows) if total_rows and total_rows > 0 else None
            self._relevant = None
            self._relevant_label = relevant_label
            self._activity = activity
        self._render()

    def update(
        self,
        rows: int | None = None,
        *,
        total_rows: int | None = None,
        fraction: float | None = None,
        relevant: int | None = None,
        activity: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            if rows is not None:
                self._rows = rows
            if total_rows is not None:
                self._total_rows = total_rows
            if fraction is not None:
                self._fraction = max(0.0, min(1.0, float(fraction)))
            elif self._total_rows and self._total_rows > 0:
                self._fraction = max(0.0, min(1.0, self._rows / self._total_rows))
            if relevant is not None:
                self._relevant = relevant
            if activity is not None:
                self._activity = activity

    def finish_stage(self, rows: int, *, relevant: int | None = None) -> None:
        if not self.enabled:
            return
        self.update(rows, fraction=1.0, relevant=relevant, activity="complete")
        self._render(final=True)
        with self._lock:
            self._stage_name = ""

    def skip_stage(self, name: str, detail: str = "already complete") -> None:
        if not self.enabled:
            return
        self._clear_line()
        print(
            f"[{self._dataset_index}/{self.dataset_total}] {self._dataset_name} | "
            f"{name}: {detail}",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._clear_line()

    def _refresh_loop(self) -> None:
        while not self._stop.wait(1.0):
            self._render()

    def _clear_line(self) -> None:
        if not self.enabled:
            return
        sys.stderr.write("\r\x1b[2K")
        sys.stderr.flush()

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None or seconds < 0 or not np.isfinite(seconds):
            return "--"
        seconds = int(seconds)
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _render(self, final: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._stage_name:
                return
            now = time.monotonic()
            elapsed = max(0.001, now - self._stage_start)
            rows_done = max(0, self._rows - self._stage_start_rows)
            rate = rows_done / elapsed
            fraction = self._fraction
            if fraction is None and self._total_rows:
                fraction = self._rows / self._total_rows
            fraction = None if fraction is None else max(0.0, min(1.0, fraction))
            eta = None
            if fraction is not None and 0.002 < fraction < 1.0:
                eta = elapsed * (1.0 - fraction) / fraction
            width = 24
            if fraction is None:
                spinner = "|/-\\"[int(now * 4) % 4]
                bar = spinner + " " * (width - 1)
                percent = "  ?.?%"
            else:
                filled = width if final else min(width, int(fraction * width))
                bar = "#" * filled + "-" * (width - filled)
                percent = f"{fraction * 100:6.1f}%"
            row_text = f"{self._rows:,}"
            if self._total_rows:
                row_text += f"/{self._total_rows:,}"
            relevant_text = (
                f" | {self._relevant:,} {self._relevant_label}"
                if self._relevant is not None
                else ""
            )
            rate_text = f" | {rate:,.0f} rows/s" if rows_done else ""
            eta_text = f" | ETA {self._duration(eta)}" if eta is not None else ""
            activity = f" | {self._activity}" if self._activity else ""
            line = (
                f"[{self._dataset_index}/{self.dataset_total}] {self._dataset_name} | "
                f"{self._stage_name:<7} [{bar}] {percent} | {row_text}"
                f"{relevant_text}{rate_text} | elapsed {self._duration(elapsed)}{eta_text}{activity}"
            )
        self._clear_line()
        sys.stderr.write(line)
        if final:
            sys.stderr.write("\n")
        sys.stderr.flush()


def source_fraction(handle: Any, path: Path) -> float | None:
    """Approximate source progress from compressed bytes consumed.

    Buffered gzip reads may run slightly ahead of parsed records, so this is an ETA
    aid rather than a checkpoint value. Checkpoint correctness still uses source rows.
    """
    raw = getattr(handle, "_compressed_raw", None)
    size = getattr(handle, "_compressed_size", None)
    if raw is None or not size:
        return None
    try:
        return min(1.0, max(0.0, raw.tell() / size))
    except (AttributeError, OSError):
        return None


S3_BUCKET = "alistairbegg-personal-projects"
S3_PREFIX = "mtg_card_embeddings/data"
S3_REGION = "eu-west-2"
S3_DATABASE_KEY = f"{S3_PREFIX}/17lands.duckdb"
S3_MANIFEST_KEY = f"{S3_PREFIX}/manifest.json"
S3_HASH_CHUNK_SIZE = 8 * 1024 * 1024
GAME_KEY = ("draft_id", "match_number", "game_number")
TURN_RE = re.compile(r"^(user|oppo)_turn_(\d+)_(.+)$")
GAME_CARD_PREFIXES = ("opening_hand", "drawn", "tutored", "deck", "sideboard")
GAME_STAT_PREFIXES = ("opening_hand", "drawn", "tutored")

TOTAL_FIELDS = {
    "cards_drawn": "total_cards_drawn",
    "cards_tutored": "total_cards_tutored",
    "cards_drawn_or_tutored": "total_cards_drawn_or_tutored",
    "cards_discarded": "total_cards_discarded",
    "lands_played": "total_lands_played",
    "cards_foretold": "total_cards_foretold",
    "creatures_cast": "total_creatures_cast",
    "creatures_blitzed": "total_creatures_blitzed",
    "non_creatures_cast": "total_non_creatures_cast",
    "instants_sorceries_cast": "total_instants_sorceries_cast",
    "cards_learned": "total_cards_learned",
    "mana_spent": "total_mana_spent",
}

ZONE_FIELDS = {
    "eot_user_cards_in_hand": (True, "hand"),
    "eot_user_lands_in_play": (True, "battlefield_land"),
    "eot_oppo_lands_in_play": (False, "battlefield_land"),
    "eot_user_creatures_in_play": (True, "battlefield_creature"),
    "eot_oppo_creatures_in_play": (False, "battlefield_creature"),
    "eot_user_non_creatures_in_play": (True, "battlefield_noncreature"),
    "eot_oppo_non_creatures_in_play": (False, "battlefield_noncreature"),
}

GAME_METADATA_COLUMNS = {
    *GAME_KEY,
    "expansion",
    "event_type",
    "draft_time",
    "game_time",
    "build_index",
    "rank",
    "opp_rank",
    "main_colors",
    "splash_colors",
    "on_play",
    "num_mulligans",
    "opp_num_mulligans",
    "opp_colors",
    "num_turns",
    "won",
}

PAIR_CHECK_FIELDS = {
    "on_play": "bool",
    "won": "bool",
    "num_mulligans": "int",
    "opp_num_mulligans": "int",
    "num_turns": "int",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS drafts (
    draft_id VARCHAR PRIMARY KEY,
    expansion VARCHAR,
    event_type VARCHAR,
    draft_time TIMESTAMP,
    rank VARCHAR,
    event_match_wins SMALLINT,
    event_match_losses SMALLINT,
    user_n_games_bucket SMALLINT,
    user_game_win_rate_bucket REAL
);

CREATE TABLE IF NOT EXISTS cards (
    card_id BIGINT PRIMARY KEY,
    card_name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS arena_card_ids (
    arena_card_id BIGINT PRIMARY KEY,
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    source_json VARCHAR
);

CREATE TABLE IF NOT EXISTS abilities (
    ability_id BIGINT PRIMARY KEY,
    ability_text VARCHAR,
    source_json VARCHAR
);

CREATE TABLE IF NOT EXISTS draft_picks (
    pick_id BIGINT PRIMARY KEY,
    draft_id VARCHAR NOT NULL REFERENCES drafts(draft_id),
    pack_number SMALLINT NOT NULL,
    pick_number SMALLINT NOT NULL,
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    pick_maindeck_rate REAL,
    pick_sideboard_in_rate REAL,
    UNIQUE (draft_id, pack_number, pick_number)
);

CREATE TABLE IF NOT EXISTS draft_pick_cards (
    pick_id BIGINT NOT NULL REFERENCES draft_picks(pick_id),
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    pack_count SMALLINT NOT NULL,
    pool_count SMALLINT NOT NULL,
    PRIMARY KEY (pick_id, card_id)
);

CREATE TABLE IF NOT EXISTS deck_builds (
    build_id BIGINT PRIMARY KEY,
    draft_id VARCHAR NOT NULL REFERENCES drafts(draft_id),
    build_index SMALLINT NOT NULL,
    UNIQUE (draft_id, build_index)
);

CREATE TABLE IF NOT EXISTS deck_build_cards (
    build_id BIGINT NOT NULL REFERENCES deck_builds(build_id),
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    deck_count SMALLINT NOT NULL,
    sideboard_count SMALLINT NOT NULL,
    PRIMARY KEY (build_id, card_id)
);

CREATE TABLE IF NOT EXISTS games (
    game_id BIGINT PRIMARY KEY,
    draft_id VARCHAR NOT NULL REFERENCES drafts(draft_id),
    build_id BIGINT REFERENCES deck_builds(build_id),
    match_number SMALLINT NOT NULL,
    game_number SMALLINT NOT NULL,
    game_time TIMESTAMP,
    user_on_play BOOLEAN,
    user_won BOOLEAN,
    source_num_turns SMALLINT,
    UNIQUE (draft_id, match_number, game_number)
);

CREATE TABLE IF NOT EXISTS game_players (
    game_id BIGINT NOT NULL REFERENCES games(game_id),
    is_user BOOLEAN NOT NULL,
    rank VARCHAR,
    main_colors VARCHAR,
    splash_colors VARCHAR,
    observed_colors VARCHAR,
    num_mulligans SMALLINT,
    n_games_bucket SMALLINT,
    game_win_rate_bucket REAL,
    PRIMARY KEY (game_id, is_user)
);

CREATE TABLE IF NOT EXISTS game_card_stats (
    game_id BIGINT NOT NULL REFERENCES games(game_id),
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    opening_hand_count SMALLINT NOT NULL,
    drawn_count SMALLINT NOT NULL,
    tutored_count SMALLINT NOT NULL,
    PRIMARY KEY (game_id, card_id)
);

CREATE TABLE IF NOT EXISTS candidate_hands (
    hand_id BIGINT PRIMARY KEY,
    game_id BIGINT NOT NULL REFERENCES games(game_id),
    attempt_number SMALLINT NOT NULL,
    is_final_candidate BOOLEAN NOT NULL,
    UNIQUE (game_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS candidate_hand_cards (
    hand_id BIGINT NOT NULL REFERENCES candidate_hands(hand_id),
    slot_number SMALLINT NOT NULL,
    source_arena_card_id BIGINT,
    card_id BIGINT REFERENCES cards(card_id),
    PRIMARY KEY (hand_id, slot_number)
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id BIGINT PRIMARY KEY,
    game_id BIGINT NOT NULL REFERENCES games(game_id),
    is_user_turn BOOLEAN NOT NULL,
    source_turn_index SMALLINT NOT NULL,
    UNIQUE (game_id, is_user_turn, source_turn_index)
);

CREATE TABLE IF NOT EXISTS events (
    event_id BIGINT PRIMARY KEY,
    game_id BIGINT NOT NULL REFERENCES games(game_id),
    source_turn_id BIGINT NOT NULL REFERENCES turns(turn_id),
    actual_turn_id BIGINT REFERENCES turns(turn_id),
    event_type VARCHAR NOT NULL,
    actor_is_user BOOLEAN,
    affected_is_user BOOLEAN,
    source_arena_card_id BIGINT,
    card_id BIGINT REFERENCES cards(card_id),
    source_ability_id BIGINT,
    ability_id BIGINT REFERENCES abilities(ability_id),
    numeric_value DOUBLE,
    source_ordinal SMALLINT NOT NULL,
    source_field VARCHAR NOT NULL,
    UNIQUE (game_id, source_field, source_ordinal)
);

CREATE TABLE IF NOT EXISTS turn_player_state (
    turn_id BIGINT NOT NULL REFERENCES turns(turn_id),
    player_is_user BOOLEAN NOT NULL,
    life REAL,
    poison_counters REAL,
    hand_size SMALLINT,
    mana_spent REAL,
    PRIMARY KEY (turn_id, player_is_user)
);

CREATE TABLE IF NOT EXISTS turn_zone_cards (
    turn_id BIGINT NOT NULL REFERENCES turns(turn_id),
    owner_is_user BOOLEAN NOT NULL,
    zone VARCHAR NOT NULL,
    source_arena_card_id BIGINT NOT NULL,
    card_id BIGINT REFERENCES cards(card_id),
    quantity SMALLINT NOT NULL,
    PRIMARY KEY (turn_id, owner_is_user, zone, source_arena_card_id)
);

CREATE TABLE IF NOT EXISTS game_player_totals (
    game_id BIGINT NOT NULL REFERENCES games(game_id),
    is_user BOOLEAN NOT NULL,
    cards_drawn SMALLINT,
    cards_tutored SMALLINT,
    cards_drawn_or_tutored SMALLINT,
    cards_discarded SMALLINT,
    lands_played SMALLINT,
    cards_foretold SMALLINT,
    creatures_cast SMALLINT,
    creatures_blitzed SMALLINT,
    non_creatures_cast SMALLINT,
    instants_sorceries_cast SMALLINT,
    cards_learned SMALLINT,
    mana_spent REAL,
    PRIMARY KEY (game_id, is_user)
);

CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
    dataset_id VARCHAR PRIMARY KEY,
    draft_file VARCHAR NOT NULL,
    game_file VARCHAR NOT NULL,
    replay_file VARCHAR NOT NULL,
    draft_signature VARCHAR NOT NULL,
    game_signature VARCHAR NOT NULL,
    replay_signature VARCHAR NOT NULL,
    next_draft_index BIGINT NOT NULL,
    committed_drafts BIGINT NOT NULL,
    completed BOOLEAN NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS source_ingestion_checkpoints (
    dataset_id VARCHAR NOT NULL,
    phase VARCHAR NOT NULL,
    source_file VARCHAR NOT NULL,
    source_signature VARCHAR NOT NULL,
    next_source_row BIGINT NOT NULL,
    completed BOOLEAN NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (dataset_id, phase)
);

CREATE TABLE IF NOT EXISTS ingestion_draft_status (
    dataset_id VARCHAR NOT NULL,
    draft_index BIGINT NOT NULL,
    draft_id VARCHAR NOT NULL,
    expected_games BIGINT NOT NULL,
    replay_games_processed BIGINT NOT NULL,
    completed BOOLEAN NOT NULL,
    PRIMARY KEY (dataset_id, draft_index),
    UNIQUE (dataset_id, draft_id)
);
"""


PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"

DRAFT_REQUIRED_COLUMNS = {
    "draft_id",
    "expansion",
    "event_type",
    "draft_time",
    "pack_number",
    "pick_number",
    "pick",
}


def make_pick_id(draft_id: str, pack_number: int, pick_number: int) -> int:
    return deterministic_bigint("pick", draft_id, pack_number, pick_number)


def parse_draft_card_columns(columns: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    pack: dict[str, str] = {}
    pool: dict[str, str] = {}
    for column in columns:
        if column.startswith(PACK_PREFIX):
            pack[column] = normalize_card_name(column[len(PACK_PREFIX):])
        elif column.startswith(POOL_PREFIX):
            pool[column] = normalize_card_name(column[len(POOL_PREFIX):])
    return pack, pool


def source_count(value: str) -> int:
    # csv.reader already returns strings. The overwhelming majority of these wide
    # count columns are exactly "0", "1", or "2"; keep that hot path allocation-free.
    if value == "" or value == "0" or value == "0.0":
        return 0
    if value == "1":
        return 1
    if value == "2":
        return 2
    if value == "3":
        return 3
    try:
        return int(value)
    except ValueError:
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            return 0
        return int(float(text))


def csv_field_at(record: str, target_index: int) -> str:
    """Extract a field while avoiding a full parse of a wide record.

    17Lands puts draft_id at/near the front. The common path therefore stays in
    CPython's C string routines and is effectively independent of record width.
    A quote-aware parser is used only if a quoted field occurs before the target.
    """
    if target_index == 0:
        head, sep, _tail = record.partition(",")
        if not sep:
            return head.rstrip("\r\n")
        if '"' not in head:
            return head

    start = 0
    for field_index in range(target_index + 1):
        comma = record.find(",", start)
        end = len(record) if comma < 0 else comma
        candidate = record[start:end]
        # If no quote has appeared in the prefix, normal comma positions are safe.
        if '"' not in record[:end]:
            if field_index == target_index:
                return candidate.rstrip("\r\n")
            start = end + 1
            continue
        break

    # Rare fallback for quoted commas before the requested field.
    return next(csv.reader([record]))[target_index]


def parse_csv_record(record: str, expected_columns: int, source_name: str, source_row: int) -> list[str]:
    # Most 17Lands data rows contain no CSV quoting. str.split is implemented in C
    # and is substantially faster than csv.reader for ~1,000-column records.
    if '"' not in record:
        row = record.rstrip("\r\n").split(",")
    else:
        row = next(csv.reader([record]))
    if len(row) != expected_columns:
        raise ValueError(
            f"Malformed {source_name} CSV row {source_row}: {len(row)} fields, "
            f"expected {expected_columns}"
        )
    return row


def contiguous_span(fields: list[tuple[int, int]]) -> tuple[int, int, np.ndarray] | None:
    """Return [start, stop) and aligned card IDs when source positions are contiguous."""
    if not fields:
        return None
    positions = [position for position, _card_id in fields]
    start = positions[0]
    stop = start + len(positions)
    if positions != list(range(start, stop)):
        return None
    return start, stop, np.asarray([card_id for _position, card_id in fields], dtype=np.int64)


def count_vector(values: list[str]) -> np.ndarray:
    """Convert a dense run of draft count strings in compiled NumPy code.

    Normal 17Lands draft count fields are integer strings. A tolerant fallback
    preserves the old parser for unusual blanks/float spellings.
    """
    try:
        return np.asarray(values, dtype=np.int16)
    except (TypeError, ValueError):
        return np.fromiter((source_count(value) for value in values), dtype=np.int16, count=len(values))


def split_csv_prefix(record: str, field_count: int) -> tuple[list[str], str]:
    """Parse exactly field_count CSV fields and return the untouched tail.

    This is designed for draft_data where a small metadata prefix is followed by
    ~1,090 numeric count fields. The common no-quote prefix uses str.split in C;
    the quote-aware fallback scans only the short metadata prefix, not the tail.
    """
    if field_count <= 0:
        return [], record.rstrip("\r\n")

    parts = record.split(",", field_count)
    if len(parts) == field_count + 1 and not any('"' in part for part in parts[:field_count]):
        return parts[:field_count], parts[field_count].rstrip("\r\n")

    fields: list[str] = []
    chars: list[str] = []
    in_quotes = False
    i = 0
    n = len(record)
    while i < n and len(fields) < field_count:
        ch = record[i]
        if ch == '"':
            if in_quotes and i + 1 < n and record[i + 1] == '"':
                chars.append('"')
                i += 2
                continue
            in_quotes = not in_quotes
            i += 1
            continue
        if ch == "," and not in_quotes:
            fields.append("".join(chars))
            chars.clear()
            i += 1
            continue
        if ch not in "\r\n":
            chars.append(ch)
        i += 1

    if len(fields) != field_count:
        raise ValueError(f"CSV record ended before {field_count} prefix fields were parsed")
    return fields, record[i:].rstrip("\r\n")


def numeric_csv_tail(tail: str, expected_count: int) -> np.ndarray | None:
    """Parse a comma-separated numeric tail in NumPy/C, returning int16 counts.

    None means the row used an unusual spelling (for example an empty count) and
    should take the fully tolerant CSV fallback.
    """
    if expected_count == 0:
        return np.empty(0, dtype=np.int16)
    try:
        values = np.fromstring(tail, sep=",", dtype=np.float32)
    except ValueError:
        return None
    if len(values) != expected_count:
        return None
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > np.iinfo(np.int16).max):
        return None
    rounded = values.astype(np.int16)
    if not np.all(values == rounded):
        return None
    return rounded


def skip_checkpointed_records(
    records: Iterable[str],
    count: int,
    progress_every: int,
) -> int:
    """Skip already-committed records without running csv.reader on them."""
    skipped = 0
    for _ in range(count):
        try:
            next(records)
        except StopIteration as exc:
            raise ValueError(
                f"Source ended while skipping {count:,} checkpointed rows; reached {skipped:,}"
            ) from exc
        skipped += 1
        if progress_every > 0 and skipped % progress_every == 0:
            LOG.info(
                "Resume catch-up: %s/%s checkpointed source rows skipped.",
                f"{skipped:,}",
                f"{count:,}",
            )
    if count:
        LOG.info("Resume catch-up complete at source row %s.", f"{count:,}")
    return skipped


def upsert_draft_metadata(
    con: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    con.register("_draft_metadata_chunk", frame)
    try:
        mismatch = con.execute(
            """
            SELECT s.draft_id
            FROM _draft_metadata_chunk s
            JOIN drafts d USING (draft_id)
            WHERE (s.expansion IS NOT NULL AND d.expansion IS NOT NULL
                   AND d.expansion IS DISTINCT FROM s.expansion)
               OR (s.event_type IS NOT NULL AND d.event_type IS NOT NULL
                   AND d.event_type IS DISTINCT FROM s.event_type)
               OR (s.draft_time IS NOT NULL AND d.draft_time IS NOT NULL
                   AND d.draft_time IS DISTINCT FROM CAST(s.draft_time AS TIMESTAMP))
               OR (s.rank IS NOT NULL AND d.rank IS NOT NULL
                   AND d.rank IS DISTINCT FROM s.rank)
               OR (s.event_match_wins IS NOT NULL AND d.event_match_wins IS NOT NULL
                   AND d.event_match_wins IS DISTINCT FROM CAST(s.event_match_wins AS SMALLINT))
               OR (s.event_match_losses IS NOT NULL AND d.event_match_losses IS NOT NULL
                   AND d.event_match_losses IS DISTINCT FROM CAST(s.event_match_losses AS SMALLINT))
               OR (s.user_n_games_bucket IS NOT NULL AND d.user_n_games_bucket IS NOT NULL
                   AND d.user_n_games_bucket IS DISTINCT FROM CAST(s.user_n_games_bucket AS SMALLINT))
               OR (s.user_game_win_rate_bucket IS NOT NULL AND d.user_game_win_rate_bucket IS NOT NULL
                   AND CAST(d.user_game_win_rate_bucket AS REAL)
                       IS DISTINCT FROM CAST(s.user_game_win_rate_bucket AS REAL))
            LIMIT 10
            """
        ).fetchall()
        if mismatch:
            raise ValueError(
                f"Draft metadata differs from existing database for draft IDs: "
                f"{[str(row[0]) for row in mismatch]}"
            )
        con.execute(
            """
            UPDATE drafts AS d
            SET expansion = coalesce(d.expansion, s.expansion),
                event_type = coalesce(d.event_type, s.event_type),
                draft_time = coalesce(d.draft_time, CAST(s.draft_time AS TIMESTAMP)),
                rank = coalesce(d.rank, s.rank),
                event_match_wins = coalesce(d.event_match_wins, CAST(s.event_match_wins AS SMALLINT)),
                event_match_losses = coalesce(d.event_match_losses, CAST(s.event_match_losses AS SMALLINT)),
                user_n_games_bucket = coalesce(d.user_n_games_bucket, CAST(s.user_n_games_bucket AS SMALLINT)),
                user_game_win_rate_bucket = coalesce(
                    d.user_game_win_rate_bucket, CAST(s.user_game_win_rate_bucket AS REAL)
                )
            FROM _draft_metadata_chunk AS s
            WHERE d.draft_id = s.draft_id
            """
        )
        con.execute(
            """
            INSERT INTO drafts (
                draft_id, expansion, event_type, draft_time, rank, event_match_wins,
                event_match_losses, user_n_games_bucket, user_game_win_rate_bucket
            )
            SELECT
                s.draft_id, s.expansion, s.event_type, CAST(s.draft_time AS TIMESTAMP), s.rank,
                CAST(s.event_match_wins AS SMALLINT), CAST(s.event_match_losses AS SMALLINT),
                CAST(s.user_n_games_bucket AS SMALLINT), CAST(s.user_game_win_rate_bucket AS REAL)
            FROM _draft_metadata_chunk s
            LEFT JOIN drafts d USING (draft_id)
            WHERE d.draft_id IS NULL
            """
        )
    finally:
        con.unregister("_draft_metadata_chunk")



@dataclass(frozen=True)
class EventRule:
    event_type: str
    payload: str
    actor: str
    affected: str | None = None
    exact_turn: bool = False


@dataclass(frozen=True)
class Paths:
    dataset_id: str
    draft_file: Path
    game_file: Path
    replay_file: Path
    cards_file: Path
    abilities_file: Path
    output: Path


@dataclass(frozen=True)
class ReferenceData:
    card_name_to_id: dict[str, int]
    arena_to_card_id: dict[int, int]
    ability_ids: set[int]


@dataclass(frozen=True)
class Checkpoint:
    next_draft_index: int
    committed_drafts: int
    completed: bool


@dataclass(frozen=True)
class SourceCheckpoint:
    next_source_row: int
    completed: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(S3_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class S3BackupManager:
    """Periodically upload committed DuckDB snapshots when explicitly enabled.

    Uploads are synchronous and are only attempted after a source chunk has committed.
    The active DuckDB connection is checkpointed before the file is read, so the uploaded
    .duckdb object is a self-contained snapshot. No boto3 import, AWS credential lookup,
    checkpoint-for-upload, checksum, or network call occurs when this manager is absent.
    """

    def __init__(self, db_path: Path, interval_minutes: float) -> None:
        if interval_minutes <= 0:
            raise ValueError("--s3-backup-every-minutes must be positive")
        self.db_path = db_path
        self.interval_seconds = interval_minutes * 60.0
        self.last_upload_monotonic = time.monotonic()
        self.last_attempt_monotonic = 0.0
        self.s3 = None

    def _client(self):
        if self.s3 is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "S3 backups require boto3/botocore. Install them or omit "
                    "--s3-backup-every-minutes."
                ) from exc
            session = boto3.Session()
            self.s3 = session.client(
                "s3",
                region_name=S3_REGION,
                config=Config(signature_version="s3v4"),
            )
        return self.s3

    def _manifest(self) -> dict[str, Any]:
        return {
            "database": self.db_path.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": self.db_path.stat().st_size,
            "sha256": sha256_file(self.db_path),
            "git_commit": get_git_commit(),
        }

    def maybe_upload(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        reason: str,
        force: bool = False,
    ) -> bool:
        now = time.monotonic()
        if not force and now - self.last_upload_monotonic < self.interval_seconds:
            return False

        # If AWS is temporarily unavailable, avoid retrying on every tiny DB commit.
        retry_floor = min(self.interval_seconds, 300.0)
        if not force and self.last_attempt_monotonic and now - self.last_attempt_monotonic < retry_floor:
            return False
        self.last_attempt_monotonic = now

        started = time.monotonic()
        LOG.info(
            "S3 backup due (%s): checkpointing %s before upload.",
            reason,
            self.db_path,
        )
        try:
            con.execute("CHECKPOINT")
            manifest = self._manifest()
            s3 = self._client()
            LOG.info(
                "S3 backup: uploading %s bytes to s3://%s/%s.",
                f"{manifest['size_bytes']:,}",
                S3_BUCKET,
                S3_DATABASE_KEY,
            )
            s3.upload_file(str(self.db_path), S3_BUCKET, S3_DATABASE_KEY)
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=S3_MANIFEST_KEY,
                Body=json.dumps(manifest, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
        except Exception:
            LOG.exception(
                "S3 backup failed; ingestion will continue and retry after a later commit."
            )
            return False

        self.last_upload_monotonic = time.monotonic()
        LOG.info(
            "S3 backup complete in %.1fs: s3://%s/%s",
            time.monotonic() - started,
            S3_BUCKET,
            S3_DATABASE_KEY,
        )
        return True


class DatabaseBuilder:
    def __init__(
        self,
        paths: Paths,
        batch_size: int,
        overwrite: bool,
        progress_every: int,
        s3_backup: S3BackupManager | None = None,
        progress: ProgressDisplay | None = None,
        dataset_position: int = 1,
    ) -> None:
        self.paths = paths
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.progress_every = progress_every
        self.s3_backup = s3_backup
        self.progress = progress
        self.dataset_position = dataset_position

        self.draft_columns = read_header(paths.draft_file)
        self.game_columns = read_header(paths.game_file)
        self.replay_columns = read_header(paths.replay_file)
        self.draft_pack_columns, self.draft_pool_columns = parse_draft_card_columns(self.draft_columns)
        self.game_card_columns = parse_game_card_columns(self.game_columns)
        self.turn_columns = parse_turn_columns(self.replay_columns)
        self.event_columns = {
            column: rule
            for column, (source_side, _, suffix) in self.turn_columns.items()
            if (rule := classify_event_suffix(source_side, suffix)) is not None
        }
        self.candidate_columns = sorted(
            [column for column in self.replay_columns if re.fullmatch(r"candidate_hand_\d+", column)],
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        self.reference: ReferenceData | None = None

    def run(self) -> None:
        if self.progress is not None:
            self.progress.begin_dataset(self.dataset_position, self.paths.dataset_id)
        if self.batch_size <= 0:
            raise ValueError("--batch-size must be positive")

        require_columns(self.draft_columns, DRAFT_REQUIRED_COLUMNS, "draft CSV")
        validate_required_columns(self.game_columns, self.replay_columns, self.candidate_columns)
        validate_turn_schema(self.turn_columns)
        validate_repeated_turn_schema(self.turn_columns)
        prepare_output(self.paths.output, self.overwrite)

        helper_cards = pd.read_csv(self.paths.cards_file)
        helper_abilities = pd.read_csv(self.paths.abilities_file)
        require_columns(helper_cards.columns, {"id", "name"}, self.paths.cards_file.name)
        require_columns(helper_abilities.columns, {"id", "text"}, self.paths.abilities_file.name)

        draft_card_names = set(self.draft_pack_columns.values()) | set(self.draft_pool_columns.values())
        self.reference = build_reference_data(
            helper_cards,
            helper_abilities,
            self.game_card_columns,
            draft_card_names,
        )

        draft_signature = file_signature(self.paths.draft_file, self.draft_columns)
        game_signature = file_signature(self.paths.game_file, self.game_columns)
        replay_signature = file_signature(self.paths.replay_file, self.replay_columns)

        con = duckdb.connect(str(self.paths.output))
        try:
            con.execute(SCHEMA_SQL)
            validate_database_schema(con)
            self.sync_reference_tables(con, helper_cards, helper_abilities)
            checkpoint = self.load_or_create_checkpoint(
                con,
                draft_signature,
                game_signature,
                replay_signature,
            )

            if checkpoint.completed:
                if self.progress is not None and self.progress.enabled:
                    self.progress.skip_stage("Dataset", "already complete")
                else:
                    LOG.info("Dataset %s is already complete.", self.paths.dataset_id)
                return

            # Preserve the existing draft-oriented checkpoint exactly.  The source
            # phase checkpoints below are additive and make each large gzip scan
            # resumable without changing the meaning of next_draft_index.
            self.build_draft_order(con, game_signature, checkpoint)
            total_drafts = int(con.execute(
                "SELECT count(*) FROM ingestion_draft_status WHERE dataset_id = ?",
                [self.paths.dataset_id],
            ).fetchone()[0])
            if checkpoint.next_draft_index > total_drafts:
                raise ValueError(
                    f"Checkpoint draft index {checkpoint.next_draft_index} exceeds "
                    f"the {total_drafts} drafts found in game_data"
                )

            LOG.info(
                "Resuming %s at draft %d/%d. Source files will each be scanned once.",
                self.paths.dataset_id,
                checkpoint.next_draft_index,
                total_drafts,
            )

            self.ingest_drafts_single_pass(con, draft_signature)
            self.ingest_games_single_pass(con, game_signature)
            self.ingest_replays_single_pass(con, replay_signature, total_drafts)

            final_checkpoint = Checkpoint(total_drafts, total_drafts, False)
            self.mark_complete(con, final_checkpoint)
            if self.progress is None or not self.progress.enabled:
                LOG.info("Dataset %s is complete.", self.paths.dataset_id)
        finally:
            con.close()

    def maybe_s3_backup(self, con: duckdb.DuckDBPyConnection, reason: str) -> None:
        if self.s3_backup is not None:
            self.s3_backup.maybe_upload(con, reason=reason)

    def game_usecols(self) -> list[str]:
        wanted = set(GAME_METADATA_COLUMNS)
        wanted.update(item["column"] for item in self.game_card_columns)
        return [column for column in self.game_columns if column in wanted]

    def replay_usecols(self) -> list[str]:
        wanted = set(GAME_KEY)
        wanted.update({
            "on_play",
            "won",
            "num_turns",
            "num_mulligans",
            "opp_num_mulligans",
            "opening_hand",
            "user_n_games_bucket",
            "user_game_win_rate_bucket",
        })
        wanted.update(self.candidate_columns)
        wanted.update(self.turn_columns)
        for side in ("user", "oppo"):
            for suffix in TOTAL_FIELDS.values():
                wanted.add(f"{side}_{suffix}")
        return [column for column in self.replay_columns if column in wanted]

    def load_or_create_source_checkpoint(
        self,
        con: duckdb.DuckDBPyConnection,
        phase: str,
        source_file: Path,
        source_signature: str,
    ) -> SourceCheckpoint:
        row = con.execute(
            """
            SELECT source_signature, next_source_row, completed
            FROM source_ingestion_checkpoints
            WHERE dataset_id = ? AND phase = ?
            """,
            [self.paths.dataset_id, phase],
        ).fetchone()
        if row is None:
            con.execute(
                """
                INSERT INTO source_ingestion_checkpoints (
                    dataset_id, phase, source_file, source_signature,
                    next_source_row, completed, updated_at
                ) VALUES (?, ?, ?, ?, 0, FALSE, current_timestamp)
                """,
                [self.paths.dataset_id, phase, str(source_file), source_signature],
            )
            return SourceCheckpoint(0, False)
        saved_signature, next_source_row, completed = row
        if saved_signature != source_signature:
            raise ValueError(
                f"Source checkpoint for phase {phase!r} does not match the current file. "
                "Use the original source files or rebuild with --overwrite."
            )
        return SourceCheckpoint(int(next_source_row), bool(completed))

    def build_draft_order(
        self,
        con: duckdb.DuckDBPyConnection,
        game_signature: str,
        checkpoint: Checkpoint,
    ) -> None:
        """Build first-seen draft order/counts with minimal parsing of game_data."""
        phase = self.load_or_create_source_checkpoint(
            con, "order", self.paths.game_file, game_signature
        )
        if phase.completed:
            if self.progress is not None:
                self.progress.skip_stage("Index")
            else:
                LOG.info("Index scan already complete.")
            return

        started = time.monotonic()
        if self.progress is not None:
            self.progress.start_stage(
                "Index", start_rows=phase.next_source_row, relevant_label="drafts",
                activity="scanning game index",
            )
        else:
            LOG.info(
                "Index scan: game_data starting at source row %s.",
                f"{phase.next_source_row:,}",
            )
        existing_rows = con.execute(
            """
            SELECT draft_id, draft_index
            FROM ingestion_draft_status
            WHERE dataset_id = ?
            ORDER BY draft_index
            """,
            [self.paths.dataset_id],
        ).fetchall()
        draft_to_index = {str(draft_id): int(draft_index) for draft_id, draft_index in existing_rows}
        next_index = len(draft_to_index)
        draft_position = self.game_columns.index("draft_id")
        pending_new: list[tuple[str, int, str, bool]] = []
        pending_counts: Counter[str] = Counter()
        rows_since_commit = 0
        source_row = phase.next_source_row

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal pending_new, pending_counts, rows_since_commit
            if self.progress is not None:
                self.progress.update(next_source_row, relevant=len(draft_to_index), activity="committing index checkpoint")
            con.execute("BEGIN")
            try:
                if pending_new:
                    con.executemany(
                        """
                        INSERT INTO ingestion_draft_status (
                            dataset_id, draft_index, draft_id, expected_games,
                            replay_games_processed, completed
                        ) VALUES (?, ?, ?, 0, 0, ?)
                        """,
                        pending_new,
                    )
                if pending_counts:
                    con.executemany(
                        """
                        UPDATE ingestion_draft_status
                        SET expected_games = expected_games + ?
                        WHERE dataset_id = ? AND draft_id = ?
                        """,
                        [
                            (count, self.paths.dataset_id, draft_id)
                            for draft_id, count in pending_counts.items()
                        ],
                    )
                con.execute(
                    """
                    UPDATE source_ingestion_checkpoints
                    SET next_source_row = ?, completed = ?, source_file = ?,
                        updated_at = current_timestamp
                    WHERE dataset_id = ? AND phase = 'order'
                    """,
                    [next_source_row, complete, str(self.paths.game_file), self.paths.dataset_id],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            self.maybe_s3_backup(con, "index scan checkpoint")
            pending_new = []
            pending_counts = Counter()
            rows_since_commit = 0
            if self.progress is not None:
                self.progress.update(next_source_row, relevant=len(draft_to_index), activity="scanning game index")

        with open_csv_text(self.paths.game_file) as handle:
            header = next(csv.reader([next(handle)]))
            if header != self.game_columns:
                raise ValueError("Game header changed between startup and order scan")
            records = iter(handle)
            skip_checkpointed_records(records, phase.next_source_row, self.progress_every)
            for record in records:
                source_row += 1
                draft_id = csv_field_at(record, draft_position)
                if draft_id not in draft_to_index:
                    draft_index = next_index
                    draft_to_index[draft_id] = draft_index
                    next_index += 1
                    pending_new.append((
                        self.paths.dataset_id,
                        draft_index,
                        draft_id,
                        draft_index < checkpoint.next_draft_index,
                    ))
                pending_counts[draft_id] += 1
                rows_since_commit += 1
                if rows_since_commit >= 100_000:
                    commit_progress(source_row)
                if self.progress is not None and source_row % 5_000 == 0:
                    self.progress.update(
                        source_row, relevant=len(draft_to_index),
                        fraction=source_fraction(handle, self.paths.game_file),
                    )
                elif self.progress_every > 0 and source_row % self.progress_every == 0:
                    LOG.info(
                        "Index scan: %s game rows read; %s drafts found.",
                        f"{source_row:,}",
                        f"{len(draft_to_index):,}",
                    )

        commit_progress(source_row, complete=True)
        if self.progress is not None:
            self.progress.finish_stage(source_row, relevant=len(draft_to_index))
        else:
            LOG.info(
                "Index scan complete: %s game rows; %s drafts; %.1fs.",
                f"{source_row:,}",
                f"{len(draft_to_index):,}",
                time.monotonic() - started,
            )

    def incomplete_draft_ids(self, con: duckdb.DuckDBPyConnection) -> set[str]:
        return {
            str(draft_id)
            for (draft_id,) in con.execute(
                """
                SELECT draft_id
                FROM ingestion_draft_status
                WHERE dataset_id = ? AND completed = FALSE
                """,
                [self.paths.dataset_id],
            ).fetchall()
        }

    def ingest_drafts_single_pass(
        self,
        con: duckdb.DuckDBPyConnection,
        draft_signature: str,
    ) -> None:
        assert self.reference is not None
        phase = self.load_or_create_source_checkpoint(
            con, "draft", self.paths.draft_file, draft_signature
        )
        if phase.completed:
            if self.progress is not None:
                self.progress.skip_stage("Draft")
            else:
                LOG.info("Phase 1/3 draft scan already complete.")
            return

        started = time.monotonic()
        if self.progress is not None:
            self.progress.start_stage(
                "Draft", start_rows=phase.next_source_row, relevant_label="picks",
                activity="scanning draft data",
            )
        else:
            LOG.info(
                "Phase 1/3 draft scan: starting at source row %s.",
                f"{phase.next_source_row:,}",
            )
        target_draft_ids = self.incomplete_draft_ids(con)
        existing_incomplete_picks = int(con.execute(
            """
            SELECT count(*)
            FROM draft_picks p
            JOIN ingestion_draft_status s ON s.draft_id = p.draft_id
            WHERE s.dataset_id = ? AND s.completed = FALSE
            """,
            [self.paths.dataset_id],
        ).fetchone()[0])
        validate_existing_draft_rows = existing_incomplete_picks > 0
        if self.progress is not None and not validate_existing_draft_rows:
            self.progress.update(activity="fresh draft load; validation joins skipped")
        if not target_draft_ids:
            con.execute(
                "UPDATE source_ingestion_checkpoints SET completed = TRUE, updated_at = current_timestamp "
                "WHERE dataset_id = ? AND phase = 'draft'",
                [self.paths.dataset_id],
            )
            return

        index = {column: position for position, column in enumerate(self.draft_columns)}
        draft_position = index["draft_id"]
        pack_fields = [
            (index[column], self.reference.card_name_to_id[name])
            for column, name in self.draft_pack_columns.items()
        ]
        pool_fields = [
            (index[column], self.reference.card_name_to_id[name])
            for column, name in self.draft_pool_columns.items()
        ]
        pack_span = contiguous_span(pack_fields)
        pool_span = contiguous_span(pool_fields)
        pack_position_by_card_id = {card_id: position for position, card_id in pack_fields}

        # The 17Lands draft export has a compact metadata prefix followed by all
        # pack_card_* and pool_* count columns. If that invariant holds, parse the
        # entire numeric tail with np.fromstring instead of constructing ~1,090
        # Python strings and visiting each field in Python.
        card_positions = sorted(position for position, _card_id in (*pack_fields, *pool_fields))
        draft_tail_start: int | None = None
        draft_tail_count = 0
        if card_positions:
            candidate_start = card_positions[0]
            if (
                card_positions == list(range(candidate_start, len(self.draft_columns)))
                and len(card_positions) == len(self.draft_columns) - candidate_start
            ):
                draft_tail_start = candidate_start
                draft_tail_count = len(card_positions)

        if draft_tail_start is not None:
            if self.progress is not None:
                self.progress.update(activity=f"vectorized scan ({draft_tail_count:,} count columns)")
            else:
                LOG.info(
                    "Draft scan turbo path: %s metadata fields + %s numeric count fields.",
                    f"{draft_tail_start:,}", f"{draft_tail_count:,}",
                )
        elif pack_span is not None and pool_span is not None:
            if self.progress is not None:
                self.progress.update(activity="vectorized pack/pool scan")
            else:
                LOG.info(
                    "Draft scan vectorization: pack=%s contiguous columns; pool=%s contiguous columns.",
                    f"{len(pack_fields):,}", f"{len(pool_fields):,}",
                )
        else:
            LOG.warning("Draft card columns are not contiguous; using slower sparse fallback.")

        pack_offsets_from_tail = None
        pool_offsets_from_tail = None
        pack_card_ids_tail = None
        pool_card_ids_tail = None
        if draft_tail_start is not None:
            pack_offsets_from_tail = np.asarray(
                [position - draft_tail_start for position, _card_id in pack_fields], dtype=np.int32
            )
            pool_offsets_from_tail = np.asarray(
                [position - draft_tail_start for position, _card_id in pool_fields], dtype=np.int32
            )
            pack_card_ids_tail = np.asarray([card_id for _position, card_id in pack_fields], dtype=np.int64)
            pool_card_ids_tail = np.asarray([card_id for _position, card_id in pool_fields], dtype=np.int64)

        metadata_columns = (
            "expansion", "event_type", "draft_time", "rank", "event_match_wins",
            "event_match_losses", "user_n_games_bucket", "user_game_win_rate_bucket",
        )
        metadata_positions = {column: index.get(column) for column in metadata_columns}
        seen_metadata_raw: dict[str, tuple[str | None, ...]] = {}

        draft_rows: dict[str, dict[str, Any]] = {}
        pick_rows: list[tuple[int, str, int, int, int, float | None, float | None]] = []
        pick_card_rows: list[tuple[int, int, int, int]] = []
        source_row = phase.next_source_row
        raw_since_commit = 0
        total_picks = 0
        commit_pick_limit = max(20_000, self.batch_size * 10)

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal total_picks, raw_since_commit
            if not pick_rows and not complete and raw_since_commit < 250_000:
                return
            pending_pick_count = len(pick_rows)
            if self.progress is not None:
                self.progress.update(
                    next_source_row, relevant=total_picks + pending_pick_count,
                    activity=f"committing {pending_pick_count:,} picks",
                )
            con.execute("BEGIN")
            try:
                if draft_rows:
                    upsert_draft_metadata(con, list(draft_rows.values()))
                if pick_rows:
                    insert_or_validate_draft_rows(con, pick_rows, pick_card_rows, validate_existing=validate_existing_draft_rows)
                con.execute(
                    """
                    UPDATE source_ingestion_checkpoints
                    SET next_source_row = ?, completed = ?, source_file = ?,
                        updated_at = current_timestamp
                    WHERE dataset_id = ? AND phase = 'draft'
                    """,
                    [next_source_row, complete, str(self.paths.draft_file), self.paths.dataset_id],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            self.maybe_s3_backup(con, "draft scan checkpoint")
            total_picks += len(pick_rows)
            draft_rows.clear()
            pick_rows.clear()
            pick_card_rows.clear()
            raw_since_commit = 0
            if self.progress is not None:
                self.progress.update(next_source_row, relevant=total_picks, activity="scanning draft data")

        with open_csv_text(self.paths.draft_file) as handle:
            header = next(csv.reader([next(handle)]))
            if header != self.draft_columns:
                raise ValueError("Draft header changed between startup and source scan")
            records = iter(handle)
            skip_checkpointed_records(records, phase.next_source_row, self.progress_every)
            if self.progress is not None:
                self.progress.update(
                    phase.next_source_row, fraction=source_fraction(handle, self.paths.draft_file),
                    relevant=total_picks, activity="scanning draft data",
                )
            else:
                LOG.info("Draft scan: processing new source rows.")
            for record in records:
                source_row += 1
                raw_since_commit += 1
                if self.progress is not None and source_row % 5_000 == 0:
                    self.progress.update(
                        source_row, relevant=total_picks + len(pick_rows),
                        fraction=source_fraction(handle, self.paths.draft_file),
                    )
                draft_id = csv_field_at(record, draft_position)
                if draft_id not in target_draft_ids:
                    if raw_since_commit >= 250_000:
                        commit_progress(source_row)
                    continue

                tail_counts: np.ndarray | None = None
                if draft_tail_start is not None:
                    prefix, numeric_tail = split_csv_prefix(record, draft_tail_start)
                    tail_counts = numeric_csv_tail(numeric_tail, draft_tail_count)
                    if tail_counts is not None:
                        # Prefix positions are identical to source positions because the
                        # numeric tail starts exactly at draft_tail_start.
                        row = prefix
                    else:
                        row = parse_csv_record(record, len(self.draft_columns), "draft", source_row)
                else:
                    row = parse_csv_record(record, len(self.draft_columns), "draft", source_row)

                def draft_value(position: int | None) -> str | None:
                    if position is None:
                        return None
                    if position < len(row):
                        return row[position]
                    # This only occurs on the tolerant full-row fallback.
                    return row[position]

                metadata_raw = tuple(
                    draft_value(position)
                    for position in metadata_positions.values()
                )
                previous_raw = seen_metadata_raw.get(draft_id)
                if previous_raw is None:
                    seen_metadata_raw[draft_id] = metadata_raw
                    draft_rows[draft_id] = {
                        "draft_id": draft_id,
                        "expansion": text_or_none(metadata_raw[0]),
                        "event_type": text_or_none(metadata_raw[1]),
                        "draft_time": timestamp_or_none(metadata_raw[2]),
                        "rank": text_or_none(metadata_raw[3]),
                        "event_match_wins": integer_or_none(metadata_raw[4]),
                        "event_match_losses": integer_or_none(metadata_raw[5]),
                        "user_n_games_bucket": integer_or_none(metadata_raw[6]),
                        "user_game_win_rate_bucket": numeric_or_none(metadata_raw[7]),
                    }
                elif metadata_raw != previous_raw:
                    parsed = (
                        text_or_none(metadata_raw[0]), text_or_none(metadata_raw[1]),
                        timestamp_or_none(metadata_raw[2]), text_or_none(metadata_raw[3]),
                        integer_or_none(metadata_raw[4]), integer_or_none(metadata_raw[5]),
                        integer_or_none(metadata_raw[6]), numeric_or_none(metadata_raw[7]),
                    )
                    prior = (
                        text_or_none(previous_raw[0]), text_or_none(previous_raw[1]),
                        timestamp_or_none(previous_raw[2]), text_or_none(previous_raw[3]),
                        integer_or_none(previous_raw[4]), integer_or_none(previous_raw[5]),
                        integer_or_none(previous_raw[6]), numeric_or_none(previous_raw[7]),
                    )
                    if not values_equal(prior, parsed):
                        raise ValueError(f"Conflicting draft metadata within draft {draft_id}")

                pack_number = parse_key_integer(draft_value(index["pack_number"]))
                pick_number = parse_key_integer(draft_value(index["pick_number"]))
                picked_name = normalize_card_name(draft_value(index["pick"]))
                picked_id = self.reference.card_name_to_id[picked_name]
                pick_id = make_pick_id(draft_id, pack_number, pick_number)

                offered_position = pack_position_by_card_id.get(picked_id)
                if offered_position is None:
                    LOG.warning(
                        "Source inconsistency: selected card %r has no offered-pack column "
                        "for draft %s pack=%s pick=%s; preserving source values.",
                        picked_name,
                        draft_id,
                        pack_number,
                        pick_number,
                    )

                card_counts: dict[int, list[int]] = {}
                if tail_counts is not None:
                    assert draft_tail_start is not None
                    assert pack_offsets_from_tail is not None and pool_offsets_from_tail is not None
                    assert pack_card_ids_tail is not None and pool_card_ids_tail is not None
                    pack_counts = tail_counts[pack_offsets_from_tail]
                    pool_counts = tail_counts[pool_offsets_from_tail]
                    if offered_position is not None:
                        offered_offset = offered_position - draft_tail_start
                        if offered_offset < 0 or offered_offset >= len(tail_counts) or int(tail_counts[offered_offset]) <= 0:
                            LOG.warning(
                                "Source inconsistency: selected card %r not present in offered pack "
                                "for draft %s pack=%s pick=%s; preserving source values.",
                                picked_name,
                                draft_id,
                                pack_number,
                                pick_number,
                            )
                    for offset_raw in np.flatnonzero(pack_counts):
                        offset = int(offset_raw)
                        card_counts[int(pack_card_ids_tail[offset])] = [int(pack_counts[offset]), 0]
                    for offset_raw in np.flatnonzero(pool_counts):
                        offset = int(offset_raw)
                        card_id = int(pool_card_ids_tail[offset])
                        count = int(pool_counts[offset])
                        counts = card_counts.get(card_id)
                        if counts is None:
                            card_counts[card_id] = [0, count]
                        else:
                            counts[1] = count
                elif len(row) == len(self.draft_columns) and pack_span is not None and pool_span is not None:
                    pack_start, pack_stop, pack_card_ids = pack_span
                    pool_start, pool_stop, pool_card_ids = pool_span
                    pack_counts = count_vector(row[pack_start:pack_stop])
                    pool_counts = count_vector(row[pool_start:pool_stop])
                    if offered_position is not None:
                        offered_offset = offered_position - pack_start
                        if (
                                offered_offset < 0
                                or offered_offset >= len(pack_counts)
                                or int(pack_counts[offered_offset]) <= 0
                        ):
                            LOG.warning(
                                "Source inconsistency: selected card %r not present in offered pack "
                                "for draft %s pack=%s pick=%s; preserving source values.",
                                picked_name,
                                draft_id,
                                pack_number,
                                pick_number,
                            )
                    for offset_raw in np.flatnonzero(pack_counts):
                        offset = int(offset_raw)
                        card_counts[int(pack_card_ids[offset])] = [int(pack_counts[offset]), 0]
                    for offset_raw in np.flatnonzero(pool_counts):
                        offset = int(offset_raw)
                        card_id = int(pool_card_ids[offset])
                        count = int(pool_counts[offset])
                        counts = card_counts.get(card_id)
                        if counts is None:
                            card_counts[card_id] = [0, count]
                        else:
                            counts[1] = count
                else:
                    if offered_position is not None and source_count(row[offered_position]) <= 0:
                        LOG.warning(
                            "Source inconsistency: selected card %r not present in offered pack "
                            "for draft %s pack=%s pick=%s; preserving source values.",
                            picked_name,
                            draft_id,
                            pack_number,
                            pick_number,
                        )
                    for position, card_id in pack_fields:
                        count = source_count(row[position])
                        if count:
                            card_counts[card_id] = [count, 0]
                    for position, card_id in pool_fields:
                        count = source_count(row[position])
                        if count:
                            counts = card_counts.get(card_id)
                            if counts is None:
                                card_counts[card_id] = [0, count]
                            else:
                                counts[1] = count

                pick_rows.append((
                    pick_id,
                    draft_id,
                    pack_number,
                    pick_number,
                    picked_id,
                    numeric_or_none(draft_value(index["pick_maindeck_rate"]))
                    if "pick_maindeck_rate" in index else None,
                    numeric_or_none(draft_value(index["pick_sideboard_in_rate"]))
                    if "pick_sideboard_in_rate" in index else None,
                ))
                pick_card_rows.extend(
                    (pick_id, card_id, counts[0], counts[1])
                    for card_id, counts in card_counts.items()
                )

                if len(pick_rows) >= commit_pick_limit:
                    commit_progress(source_row)
                if self.progress is None and self.progress_every > 0 and source_row % self.progress_every == 0:
                    LOG.info(
                        "Draft scan: %s rows read; %s relevant picks processed.",
                        f"{source_row:,}",
                        f"{total_picks + len(pick_rows):,}",
                    )

        commit_progress(source_row, complete=True)
        missing = int(con.execute(
            """
            SELECT count(*)
            FROM ingestion_draft_status s
            LEFT JOIN draft_picks p ON p.draft_id = s.draft_id
            WHERE s.dataset_id = ? AND s.completed = FALSE AND p.draft_id IS NULL
            """,
            [self.paths.dataset_id],
        ).fetchone()[0])
        if missing:
            raise ValueError(f"Draft source completed but {missing} remaining drafts have no picks")
        con.execute("CHECKPOINT")
        if self.progress is not None:
            self.progress.finish_stage(source_row, relevant=total_picks)
        else:
            LOG.info(
                "Phase 1/3 draft scan complete: %s rows; %s relevant picks; %.1fs.",
                f"{source_row:,}",
                f"{total_picks:,}",
                time.monotonic() - started,
            )

    def build_game_player_rows_without_replay(
        self,
        game_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> list[dict[str, Any]]:
        metadata = [
            column for column in (
                "rank", "main_colors", "splash_colors", "num_mulligans",
                "opp_rank", "opp_colors", "opp_num_mulligans",
            )
            if column in game_batch.columns
        ]
        records = game_batch[metadata].to_dict("records")
        rows: list[dict[str, Any]] = []
        for position, game in enumerate(records):
            game_id = game_ids[position]
            rows.append({
                "game_id": game_id, "is_user": True,
                "rank": text_or_none(game.get("rank")),
                "main_colors": text_or_none(game.get("main_colors")),
                "splash_colors": text_or_none(game.get("splash_colors")),
                "observed_colors": None,
                "num_mulligans": integer_or_none(game.get("num_mulligans")),
                "n_games_bucket": None, "game_win_rate_bucket": None,
            })
            rows.append({
                "game_id": game_id, "is_user": False,
                "rank": text_or_none(game.get("opp_rank")),
                "main_colors": None, "splash_colors": None,
                "observed_colors": text_or_none(game.get("opp_colors")),
                "num_mulligans": integer_or_none(game.get("opp_num_mulligans")),
                "n_games_bucket": None, "game_win_rate_bucket": None,
            })
        return rows

    def ingest_games_single_pass(
        self,
        con: duckdb.DuckDBPyConnection,
        game_signature: str,
    ) -> None:
        assert self.reference is not None
        phase = self.load_or_create_source_checkpoint(
            con, "game", self.paths.game_file, game_signature
        )
        if phase.completed:
            if self.progress is not None:
                self.progress.skip_stage("Games")
            else:
                LOG.info("Phase 2/3 game scan already complete.")
            return

        started = time.monotonic()
        source_total_rows = int(con.execute(
            "SELECT coalesce(sum(expected_games), 0) FROM ingestion_draft_status WHERE dataset_id = ?",
            [self.paths.dataset_id],
        ).fetchone()[0])
        if self.progress is not None:
            self.progress.start_stage(
                "Games", start_rows=phase.next_source_row, total_rows=source_total_rows,
                relevant_label="games", activity="scanning game data",
            )
        else:
            LOG.info(
                "Phase 2/3 game scan: starting at source row %s.",
                f"{phase.next_source_row:,}",
            )
        target_draft_ids = self.incomplete_draft_ids(con)
        wanted_columns = self.game_usecols()
        index = {column: position for position, column in enumerate(self.game_columns)}
        wanted_indices = [index[column] for column in wanted_columns]
        draft_position = index["draft_id"]
        rows: list[list[str]] = []
        source_row = phase.next_source_row
        raw_since_commit = 0
        total_games = 0
        chunk_limit = max(10_000, self.batch_size * 5)

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal total_games, raw_since_commit
            if not rows and not complete and raw_since_commit < 250_000:
                return
            pending_games = len(rows)
            if self.progress is not None:
                self.progress.update(
                    next_source_row, total_rows=source_total_rows,
                    relevant=total_games + pending_games,
                    activity=f"normalizing + committing {pending_games:,} games",
                )
            game_batch = pd.DataFrame(rows, columns=wanted_columns) if rows else None
            con.execute("BEGIN")
            try:
                if game_batch is not None and len(game_batch):
                    game_ids = make_game_ids(game_batch)
                    if len(set(game_ids)) != len(game_ids):
                        raise ValueError("Duplicate natural game key inside game source chunk")
                    draft_rows = self.build_draft_rows(game_batch)
                    build_ids, build_compositions = self.build_deck_compositions(game_batch)
                    game_rows = self.build_game_rows(game_batch, game_ids, build_ids)
                    player_rows = self.build_game_player_rows_without_replay(game_batch, game_ids)
                    game_stat_rows = self.build_game_card_stats(game_batch, game_ids)

                    self.ensure_drafts(con, draft_rows)
                    self.ensure_deck_builds(con, build_compositions)
                    insert_rows(
                        con, "games", game_rows,
                        int64=("game_id", "build_id"),
                        int16=("match_number", "game_number", "source_num_turns"),
                        boolean=("user_on_play", "user_won"),
                    )
                    insert_rows(
                        con, "game_players", player_rows,
                        int64=("game_id",),
                        int16=("num_mulligans", "n_games_bucket"),
                        boolean=("is_user",),
                    )
                    insert_rows(
                        con, "game_card_stats", game_stat_rows,
                        int64=("game_id", "card_id"),
                        int16=("opening_hand_count", "drawn_count", "tutored_count"),
                    )

                con.execute(
                    """
                    UPDATE source_ingestion_checkpoints
                    SET next_source_row = ?, completed = ?, source_file = ?,
                        updated_at = current_timestamp
                    WHERE dataset_id = ? AND phase = 'game'
                    """,
                    [next_source_row, complete, str(self.paths.game_file), self.paths.dataset_id],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            self.maybe_s3_backup(con, "game scan checkpoint")
            total_games += len(rows)
            rows.clear()
            raw_since_commit = 0
            if self.progress is not None:
                self.progress.update(
                    next_source_row, total_rows=source_total_rows, relevant=total_games,
                    activity="scanning game data",
                )

        with open_csv_text(self.paths.game_file) as handle:
            header = next(csv.reader([next(handle)]))
            if header != self.game_columns:
                raise ValueError("Game header changed between startup and source scan")
            records = iter(handle)
            skip_checkpointed_records(records, phase.next_source_row, self.progress_every)
            if self.progress is not None:
                self.progress.update(
                    phase.next_source_row, total_rows=source_total_rows, relevant=total_games,
                    activity="scanning game data",
                )
            else:
                LOG.info("Game scan: processing new source rows.")
            for record in records:
                source_row += 1
                raw_since_commit += 1
                if self.progress is not None and source_row % 5_000 == 0:
                    self.progress.update(
                        source_row, total_rows=source_total_rows, relevant=total_games + len(rows),
                    )
                draft_id = csv_field_at(record, draft_position)
                if draft_id in target_draft_ids:
                    row = parse_csv_record(record, len(self.game_columns), "game", source_row)
                    rows.append([row[position] for position in wanted_indices])
                if len(rows) >= chunk_limit or raw_since_commit >= 250_000:
                    commit_progress(source_row)
                if self.progress is None and self.progress_every > 0 and source_row % self.progress_every == 0:
                    LOG.info(
                        "Game scan: %s rows read; %s relevant games processed.",
                        f"{source_row:,}",
                        f"{total_games + len(rows):,}",
                    )

        commit_progress(source_row, complete=True)
        mismatches = con.execute(
            """
            SELECT s.draft_id, s.expected_games, count(g.game_id) AS actual_games
            FROM ingestion_draft_status s
            LEFT JOIN games g ON g.draft_id = s.draft_id
            WHERE s.dataset_id = ? AND s.completed = FALSE
            GROUP BY s.draft_id, s.expected_games
            HAVING count(g.game_id) <> s.expected_games
            LIMIT 10
            """,
            [self.paths.dataset_id],
        ).fetchall()
        if mismatches:
            raise ValueError(f"Game source counts do not match draft index: {mismatches}")
        con.execute("CHECKPOINT")
        if self.progress is not None:
            self.progress.finish_stage(source_row, relevant=total_games)
        else:
            LOG.info(
                "Phase 2/3 game scan complete: %s rows; %s relevant games; %.1fs.",
                f"{source_row:,}",
                f"{total_games:,}",
                time.monotonic() - started,
            )

    def validate_replay_chunk_against_database(
        self,
        con: duckdb.DuckDBPyConnection,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> None:
        id_frame = pd.DataFrame({"game_id": pd.array(game_ids, dtype="Int64")})
        con.register("_replay_validation_ids", id_frame)
        try:
            rows = con.execute(
                """
                SELECT g.game_id, g.user_on_play, g.user_won, g.source_num_turns,
                       u.num_mulligans, o.num_mulligans
                FROM _replay_validation_ids r
                LEFT JOIN games g ON g.game_id = r.game_id
                LEFT JOIN game_players u ON u.game_id = g.game_id AND u.is_user = TRUE
                LEFT JOIN game_players o ON o.game_id = g.game_id AND o.is_user = FALSE
                """
            ).fetchall()
        finally:
            con.unregister("_replay_validation_ids")

        expected = {
            int(game_id): (on_play, won, num_turns, user_mulls, opp_mulls)
            for game_id, on_play, won, num_turns, user_mulls, opp_mulls in rows
            if game_id is not None
        }
        arrays = {
            field: replay_batch[field].to_numpy(copy=False)
            for field in PAIR_CHECK_FIELDS
            if field in replay_batch.columns
        }
        for position, game_id in enumerate(game_ids):
            if game_id not in expected:
                raise ValueError(f"Replay row has no staged game row: game_id={game_id}")
            on_play, won, num_turns, user_mulls, opp_mulls = expected[game_id]
            checks = (
                ("on_play", parse_bool(arrays["on_play"][position]) if "on_play" in arrays else None, on_play),
                ("won", parse_bool(arrays["won"][position]) if "won" in arrays else None, won),
                ("num_turns", integer_or_none(arrays["num_turns"][position]) if "num_turns" in arrays else None, num_turns),
                ("num_mulligans", integer_or_none(arrays["num_mulligans"][position]) if "num_mulligans" in arrays else None, user_mulls),
                ("opp_num_mulligans", integer_or_none(arrays["opp_num_mulligans"][position]) if "opp_num_mulligans" in arrays else None, opp_mulls),
            )
            for field, source, stored in checks:
                if field not in arrays:
                    continue
                if source != stored:
                    raise ValueError(
                        f"Game/replay mismatch for {field} in game {game_id}: "
                        f"{stored!r} != {source!r}"
                    )

    def update_replay_user_metrics(
        self,
        con: duckdb.DuckDBPyConnection,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> None:
        rows = [
            {
                "game_id": game_ids[position],
                "n_games_bucket": integer_or_none(record.get("user_n_games_bucket")),
                "game_win_rate_bucket": numeric_or_none(record.get("user_game_win_rate_bucket")),
            }
            for position, record in enumerate(
                replay_batch[[
                    column for column in ("user_n_games_bucket", "user_game_win_rate_bucket")
                    if column in replay_batch.columns
                ]].to_dict("records")
            )
        ]
        if not rows:
            return
        frame = rows_to_frame(rows, int64=("game_id",), int16=("n_games_bucket",))
        con.register("_replay_user_metrics", frame)
        try:
            con.execute(
                """
                UPDATE game_players AS gp
                SET n_games_bucket = m.n_games_bucket,
                    game_win_rate_bucket = m.game_win_rate_bucket
                FROM _replay_user_metrics AS m
                WHERE gp.game_id = m.game_id AND gp.is_user = TRUE
                """
            )
        finally:
            con.unregister("_replay_user_metrics")

    def advance_main_checkpoint_from_status(
        self,
        con: duckdb.DuckDBPyConnection,
        total_drafts: int,
    ) -> int:
        first_incomplete = con.execute(
            """
            SELECT min(draft_index)
            FROM ingestion_draft_status
            WHERE dataset_id = ? AND completed = FALSE
            """,
            [self.paths.dataset_id],
        ).fetchone()[0]
        next_index = total_drafts if first_incomplete is None else int(first_incomplete)
        con.execute(
            """
            UPDATE ingestion_checkpoints
            SET next_draft_index = ?, committed_drafts = ?, updated_at = current_timestamp
            WHERE dataset_id = ?
            """,
            [next_index, next_index, self.paths.dataset_id],
        )
        return next_index

    def ingest_replays_single_pass(
        self,
        con: duckdb.DuckDBPyConnection,
        replay_signature: str,
        total_drafts: int,
    ) -> None:
        assert self.reference is not None
        phase = self.load_or_create_source_checkpoint(
            con, "replay", self.paths.replay_file, replay_signature
        )
        if phase.completed:
            if self.progress is not None:
                self.progress.skip_stage("Replay")
            else:
                LOG.info("Phase 3/3 replay scan already complete.")
            return

        started = time.monotonic()
        status_rows = con.execute(
            """
            SELECT draft_index, draft_id, expected_games, replay_games_processed, completed
            FROM ingestion_draft_status
            WHERE dataset_id = ?
            ORDER BY draft_index
            """,
            [self.paths.dataset_id],
        ).fetchall()
        status_by_id = {
            str(draft_id): [int(index), int(expected), int(processed), bool(completed)]
            for index, draft_id, expected, processed, completed in status_rows
        }
        completed_flags = [False] * total_drafts
        for index, _draft_id, _expected, _processed, completed in status_rows:
            completed_flags[int(index)] = bool(completed)
        current_checkpoint = 0
        while current_checkpoint < total_drafts and completed_flags[current_checkpoint]:
            current_checkpoint += 1
        last_checkpoint_logged = current_checkpoint

        replay_total_rows = sum(values[1] for values in status_by_id.values())
        if self.progress is not None:
            self.progress.start_stage(
                "Replay", start_rows=phase.next_source_row, total_rows=replay_total_rows,
                relevant_label="games",
                activity=f"normalizing replay data; drafts {current_checkpoint:,}/{total_drafts:,}",
            )
        else:
            LOG.info(
                "Phase 3/3 replay scan: starting at source row %s; drafts complete %s/%s (%.1f%%).",
                f"{phase.next_source_row:,}",
                f"{current_checkpoint:,}",
                f"{total_drafts:,}",
                100.0 * current_checkpoint / total_drafts if total_drafts else 100.0,
            )

        target_draft_ids = {draft_id for draft_id, values in status_by_id.items() if not values[3]}
        wanted_columns = self.replay_usecols()
        index = {column: position for position, column in enumerate(self.replay_columns)}
        missing_columns = [column for column in wanted_columns if column not in index]
        if missing_columns:
            raise ValueError(f"Replay CSV missing requested columns: {missing_columns[:20]}")
        wanted_indices = [index[column] for column in wanted_columns]
        draft_position = index["draft_id"]
        rows: list[list[str]] = []
        source_row = phase.next_source_row
        raw_since_commit = 0
        total_replays = 0
        chunk_limit = max(2_000, self.batch_size)

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal total_replays, raw_since_commit, current_checkpoint, last_checkpoint_logged
            if not rows and not complete and raw_since_commit < 250_000:
                return
            pending_replays = len(rows)
            if self.progress is not None:
                self.progress.update(
                    next_source_row, total_rows=replay_total_rows,
                    relevant=total_replays + pending_replays,
                    activity=f"normalizing + committing {pending_replays:,} replays; drafts {current_checkpoint:,}/{total_drafts:,}",
                )
            replay_batch = pd.DataFrame(rows, columns=wanted_columns) if rows else None
            touched_status: dict[str, tuple[int, int, bool]] = {}
            con.execute("BEGIN")
            try:
                if replay_batch is not None and len(replay_batch):
                    game_ids = make_game_ids(replay_batch)
                    if len(set(game_ids)) != len(game_ids):
                        raise ValueError("Duplicate natural game key inside replay source chunk")
                    self.validate_replay_chunk_against_database(con, replay_batch, game_ids)

                    positions_by_column = self.turn_meaningful_positions(replay_batch)
                    active_turns = self.find_turns(replay_batch, game_ids, positions_by_column)
                    turn_id_map = self.make_turn_id_map(active_turns)
                    turn_rows = self.build_turn_rows(active_turns, turn_id_map)
                    event_rows = self.build_event_rows(
                        replay_batch, game_ids, positions_by_column, turn_id_map
                    )
                    hand_rows, hand_card_rows = self.build_candidate_hand_rows(replay_batch, game_ids)
                    state_rows, zone_rows = self.build_turn_state_and_zones(
                        replay_batch, game_ids, active_turns, turn_id_map
                    )
                    total_rows = self.build_total_rows(replay_batch, game_ids)
                    validate_batch_rows(
                        game_ids, turn_rows, event_rows, hand_rows, hand_card_rows,
                        state_rows, zone_rows,
                    )

                    insert_rows(
                        con, "turns", turn_rows, int64=("turn_id", "game_id"),
                        int16=("source_turn_index",), boolean=("is_user_turn",),
                    )
                    insert_rows(
                        con, "candidate_hands", hand_rows, int64=("hand_id", "game_id"),
                        int16=("attempt_number",), boolean=("is_final_candidate",),
                    )
                    insert_rows(
                        con, "candidate_hand_cards", hand_card_rows,
                        int64=("hand_id", "source_arena_card_id", "card_id"),
                        int16=("slot_number",),
                    )
                    insert_rows(
                        con, "events", event_rows,
                        int64=(
                            "event_id", "game_id", "source_turn_id", "actual_turn_id",
                            "source_arena_card_id", "card_id", "source_ability_id", "ability_id",
                        ),
                        int16=("source_ordinal",), boolean=("actor_is_user", "affected_is_user"),
                    )
                    insert_rows(
                        con, "turn_player_state", state_rows, int64=("turn_id",),
                        int16=("hand_size",), boolean=("player_is_user",),
                    )
                    insert_rows(
                        con, "turn_zone_cards", zone_rows,
                        int64=("turn_id", "source_arena_card_id", "card_id"),
                        int16=("quantity",), boolean=("owner_is_user",),
                    )
                    insert_rows(
                        con, "game_player_totals", total_rows, int64=("game_id",),
                        int16=tuple(field for field in TOTAL_FIELDS if field != "mana_spent"),
                        boolean=("is_user",),
                    )
                    self.update_replay_user_metrics(con, replay_batch, game_ids)

                    counts = Counter(replay_batch["draft_id"].astype(str))
                    for draft_id, count in counts.items():
                        values = status_by_id[draft_id]
                        values[2] += int(count)
                        if values[2] > values[1]:
                            raise ValueError(
                                f"Replay source has more games than game_data for {draft_id}: "
                                f"{values[2]} > {values[1]}"
                            )
                        values[3] = values[2] == values[1]
                        completed_flags[values[0]] = values[3]
                        touched_status[draft_id] = (values[2], values[1], values[3])

                    if touched_status:
                        con.executemany(
                            """
                            UPDATE ingestion_draft_status
                            SET replay_games_processed = ?, completed = ?
                            WHERE dataset_id = ? AND draft_id = ?
                            """,
                            [
                                (processed, completed, self.paths.dataset_id, draft_id)
                                for draft_id, (processed, _expected, completed) in touched_status.items()
                            ],
                        )
                    while current_checkpoint < total_drafts and completed_flags[current_checkpoint]:
                        current_checkpoint += 1
                    con.execute(
                        """
                        UPDATE ingestion_checkpoints
                        SET next_draft_index = ?, committed_drafts = ?, updated_at = current_timestamp
                        WHERE dataset_id = ?
                        """,
                        [current_checkpoint, current_checkpoint, self.paths.dataset_id],
                    )

                con.execute(
                    """
                    UPDATE source_ingestion_checkpoints
                    SET next_source_row = ?, completed = ?, source_file = ?,
                        updated_at = current_timestamp
                    WHERE dataset_id = ? AND phase = 'replay'
                    """,
                    [next_source_row, complete, str(self.paths.replay_file), self.paths.dataset_id],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            self.maybe_s3_backup(con, "replay scan checkpoint")

            total_replays += len(rows)
            rows.clear()
            raw_since_commit = 0
            if self.progress is not None:
                self.progress.update(
                    next_source_row, total_rows=replay_total_rows, relevant=total_replays,
                    activity=f"scanning replay data; drafts {current_checkpoint:,}/{total_drafts:,}",
                )
            if current_checkpoint - last_checkpoint_logged >= 500 or complete:
                if self.progress is not None:
                    self.progress.update(
                        next_source_row, total_rows=replay_total_rows, relevant=total_replays,
                        activity=f"scanning replay data; drafts {current_checkpoint:,}/{total_drafts:,}",
                    )
                else:
                    LOG.info(
                        "Checkpoint: %s/%s drafts complete (%.1f%%).",
                        f"{current_checkpoint:,}",
                        f"{total_drafts:,}",
                        100.0 * current_checkpoint / total_drafts if total_drafts else 100.0,
                    )
                last_checkpoint_logged = current_checkpoint

        with open_csv_text(self.paths.replay_file) as handle:
            header = next(csv.reader([next(handle)]))
            if header != self.replay_columns:
                raise ValueError("Replay header changed between startup and source scan")
            records = iter(handle)
            skip_checkpointed_records(records, phase.next_source_row, self.progress_every)
            if self.progress is not None:
                self.progress.update(
                    phase.next_source_row, total_rows=replay_total_rows, relevant=total_replays,
                    activity=f"scanning replay data; drafts {current_checkpoint:,}/{total_drafts:,}",
                )
            else:
                LOG.info("Replay scan: processing new source rows.")
            for record in records:
                source_row += 1
                raw_since_commit += 1
                if self.progress is not None and source_row % 2_000 == 0:
                    self.progress.update(
                        source_row, total_rows=replay_total_rows, relevant=total_replays + len(rows),
                    )
                draft_id = csv_field_at(record, draft_position)
                if draft_id in target_draft_ids:
                    row = parse_csv_record(record, len(self.replay_columns), "replay", source_row)
                    rows.append([row[position] for position in wanted_indices])
                if len(rows) >= chunk_limit or raw_since_commit >= 250_000:
                    commit_progress(source_row)
                if self.progress is None and self.progress_every > 0 and source_row % self.progress_every == 0:
                    LOG.info(
                        "Replay scan: %s rows read; %s relevant games processed; drafts %s/%s.",
                        f"{source_row:,}",
                        f"{total_replays + len(rows):,}",
                        f"{current_checkpoint:,}",
                        f"{total_drafts:,}",
                    )

        commit_progress(source_row, complete=True)
        missing = [
            (draft_id, values[2], values[1])
            for draft_id, values in status_by_id.items()
            if not values[3]
        ][:10]
        if missing:
            raise ValueError(f"Replay source completed before all games were found: {missing}")
        con.execute("CHECKPOINT")
        if self.progress is not None:
            self.progress.finish_stage(source_row, relevant=total_replays)
        else:
            LOG.info(
                "Phase 3/3 replay scan complete: %s rows; %s relevant games; %.1fs.",
                f"{source_row:,}",
                f"{total_replays:,}",
                time.monotonic() - started,
            )

    def sync_reference_tables(
        self,
        con: duckdb.DuckDBPyConnection,
        helper_cards: pd.DataFrame,
        helper_abilities: pd.DataFrame,
    ) -> None:
        assert self.reference is not None

        existing_cards = {name: int(card_id) for card_id, name in con.execute(
            "SELECT card_id, card_name FROM cards"
        ).fetchall()}
        for name, card_id in self.reference.card_name_to_id.items():
            previous = existing_cards.get(name)
            if previous is not None and previous != card_id:
                raise ValueError(f"Canonical card ID mismatch for {name!r}")
        new_cards = [
            {"card_id": card_id, "card_name": name}
            for name, card_id in self.reference.card_name_to_id.items()
            if name not in existing_cards
        ]

        existing_arena = {int(arena_id): int(card_id) for arena_id, card_id in con.execute(
            "SELECT arena_card_id, card_id FROM arena_card_ids"
        ).fetchall()}
        arena_rows = []
        seen_arena: dict[int, int] = {}
        for record in helper_cards.to_dict("records"):
            arena_id = integer_or_none(record.get("id"))
            name = text_or_none(record.get("name"))
            if arena_id is None or name is None:
                continue
            card_id = self.reference.card_name_to_id[normalize_card_name(name)]
            prior_seen = seen_arena.get(arena_id)
            if prior_seen is not None and prior_seen != card_id:
                raise ValueError(f"Arena card ID {arena_id} maps to multiple canonical cards")
            seen_arena[arena_id] = card_id
            prior_db = existing_arena.get(arena_id)
            if prior_db is not None and prior_db != card_id:
                raise ValueError(f"Arena card ID {arena_id} conflicts with existing database mapping")
            if prior_db is None:
                arena_rows.append(
                    {
                        "arena_card_id": arena_id,
                        "card_id": card_id,
                        "source_json": json.dumps(clean_json_record(record), sort_keys=True),
                    }
                )

        existing_abilities = {int(value) for (value,) in con.execute(
            "SELECT ability_id FROM abilities"
        ).fetchall()}
        ability_rows = []
        seen_abilities: set[int] = set()
        for record in helper_abilities.to_dict("records"):
            ability_id = integer_or_none(record.get("id"))
            if ability_id is None or ability_id in seen_abilities:
                continue
            seen_abilities.add(ability_id)
            if ability_id not in existing_abilities:
                ability_rows.append(
                    {
                        "ability_id": ability_id,
                        "ability_text": text_or_none(record.get("text")),
                        "source_json": json.dumps(clean_json_record(record), sort_keys=True),
                    }
                )

        con.execute("BEGIN")
        try:
            insert_rows(con, "cards", new_cards, int64=("card_id",))
            insert_rows(con, "arena_card_ids", arena_rows, int64=("arena_card_id", "card_id"))
            insert_rows(con, "abilities", ability_rows, int64=("ability_id",))
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise

        LOG.info(
            "Reference sync: +%d cards, +%d Arena IDs, +%d abilities",
            len(new_cards),
            len(arena_rows),
            len(ability_rows),
        )

    def load_or_create_checkpoint(
        self,
        con: duckdb.DuckDBPyConnection,
        draft_signature: str,
        game_signature: str,
        replay_signature: str,
    ) -> Checkpoint:
        row = con.execute(
            """
            SELECT draft_signature, game_signature, replay_signature,
                   next_draft_index, committed_drafts, completed
            FROM ingestion_checkpoints
            WHERE dataset_id = ?
            """,
            [self.paths.dataset_id],
        ).fetchone()

        if row is None:
            con.execute(
                """
                INSERT INTO ingestion_checkpoints (
                    dataset_id, draft_file, game_file, replay_file,
                    draft_signature, game_signature, replay_signature,
                    next_draft_index, committed_drafts, completed, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, FALSE, current_timestamp)
                """,
                [
                    self.paths.dataset_id,
                    str(self.paths.draft_file),
                    str(self.paths.game_file),
                    str(self.paths.replay_file),
                    draft_signature,
                    game_signature,
                    replay_signature,
                ],
            )
            con.execute("CHECKPOINT")
            return Checkpoint(0, 0, False)

        saved_draft, saved_game, saved_replay, next_index, committed, completed = row
        if (saved_draft, saved_game, saved_replay) != (
            draft_signature,
            game_signature,
            replay_signature,
        ):
            raise ValueError(
                "Checkpoint file signatures do not match the current draft/game/replay files. "
                "Use the original files or rebuild with --overwrite."
            )
        return Checkpoint(int(next_index), int(committed), bool(completed))

    def process_draft(
        self,
        con: duckdb.DuckDBPyConnection,
        draft_id: str,
        game_batch: pd.DataFrame,
        replay_batch: pd.DataFrame,
        checkpoint: Checkpoint,
    ) -> None:
        """Transform and commit one draft atomically.

        The outer ingestion loop is draft-oriented. Within a draft we transform all
        of its games together so inserts remain bulk operations rather than creating
        a Pandas frame / DuckDB registration for every game and every child table.
        Every derived row still carries its game_id, so hands/turns/events remain
        linked to the exact game that produced them.
        """
        assert self.reference is not None
        game_batch = game_batch.reset_index(drop=True)
        replay_batch = replay_batch.reset_index(drop=True)

        if set(game_batch["draft_id"].astype(str)) != {draft_id}:
            raise ValueError(f"process_draft received game rows from multiple drafts for {draft_id}")
        validate_game_replay_pair(game_batch, replay_batch)

        game_ids = make_game_ids(game_batch)
        build_ids, build_compositions = self.build_deck_compositions(game_batch)
        game_rows = self.build_game_rows(game_batch, game_ids, build_ids)
        player_rows = self.build_player_rows(game_batch, replay_batch, game_ids)
        game_stat_rows = self.build_game_card_stats(game_batch, game_ids)

        active_turns = self.find_turns(replay_batch, game_ids)
        turn_rows = self.build_turn_rows(active_turns)
        event_rows = self.build_event_rows(replay_batch, game_ids)
        hand_rows, hand_card_rows = self.build_candidate_hand_rows(replay_batch, game_ids)
        state_rows, zone_rows = self.build_turn_state_and_zones(
            replay_batch, game_ids, active_turns
        )
        total_rows = self.build_total_rows(replay_batch, game_ids)

        validate_batch_rows(
            game_ids,
            turn_rows,
            event_rows,
            hand_rows,
            hand_card_rows,
            state_rows,
            zone_rows,
        )

        draft_rows = self.build_draft_rows(game_batch)
        con.execute("BEGIN")
        try:
            # Draft metadata/picks are loaded first from draft_data. Validate the
            # overlapping game_data metadata before inserting game children.
            self.ensure_drafts(con, draft_rows)
            self.ensure_deck_builds(con, build_compositions)

            insert_rows(
                con, "games", game_rows,
                int64=("game_id", "build_id"),
                int16=("match_number", "game_number", "source_num_turns"),
                boolean=("user_on_play", "user_won"),
            )
            insert_rows(
                con, "game_players", player_rows,
                int64=("game_id",),
                int16=("num_mulligans", "n_games_bucket"),
                boolean=("is_user",),
            )
            insert_rows(
                con, "game_card_stats", game_stat_rows,
                int64=("game_id", "card_id"),
                int16=("opening_hand_count", "drawn_count", "tutored_count"),
            )
            insert_rows(
                con, "turns", turn_rows,
                int64=("turn_id", "game_id"),
                int16=("source_turn_index",),
                boolean=("is_user_turn",),
            )
            insert_rows(
                con, "candidate_hands", hand_rows,
                int64=("hand_id", "game_id"),
                int16=("attempt_number",),
                boolean=("is_final_candidate",),
            )
            insert_rows(
                con, "candidate_hand_cards", hand_card_rows,
                int64=("hand_id", "source_arena_card_id", "card_id"),
                int16=("slot_number",),
            )
            insert_rows(
                con, "events", event_rows,
                int64=(
                    "event_id", "game_id", "source_turn_id", "actual_turn_id",
                    "source_arena_card_id", "card_id", "source_ability_id", "ability_id",
                ),
                int16=("source_ordinal",),
                boolean=("actor_is_user", "affected_is_user"),
            )
            insert_rows(
                con, "turn_player_state", state_rows,
                int64=("turn_id",),
                int16=("hand_size",),
                boolean=("player_is_user",),
            )
            insert_rows(
                con, "turn_zone_cards", zone_rows,
                int64=("turn_id", "source_arena_card_id", "card_id"),
                int16=("quantity",),
                boolean=("owner_is_user",),
            )
            insert_rows(
                con, "game_player_totals", total_rows,
                int64=("game_id",),
                int16=tuple(field for field in TOTAL_FIELDS if field != "mana_spent"),
                boolean=("is_user",),
            )

            con.execute(
                """
                UPDATE ingestion_checkpoints
                SET next_draft_index = ?, committed_drafts = ?,
                    draft_file = ?, game_file = ?, replay_file = ?,
                    updated_at = current_timestamp
                WHERE dataset_id = ?
                """,
                [
                    checkpoint.next_draft_index + 1,
                    checkpoint.committed_drafts + 1,
                    str(self.paths.draft_file),
                    str(self.paths.game_file),
                    str(self.paths.replay_file),
                    self.paths.dataset_id,
                ],
            )
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise


    def build_draft_rows(self, game_batch: pd.DataFrame) -> dict[str, dict[str, Any]]:
        columns = [column for column in ("draft_id", "expansion", "event_type", "draft_time") if column in game_batch.columns]
        result: dict[str, dict[str, Any]] = {}
        for record in game_batch[columns].to_dict("records"):
            draft_id = str(record["draft_id"])
            row = {
                "draft_id": draft_id,
                "expansion": text_or_none(record.get("expansion")),
                "event_type": text_or_none(record.get("event_type")),
                "draft_time": timestamp_or_none(record.get("draft_time")),
            }
            previous = result.get(draft_id)
            if previous is not None and previous != row:
                raise ValueError(f"Conflicting draft metadata within batch for {draft_id}")
            result[draft_id] = row
        return result

    def ensure_drafts(
        self,
        con: duckdb.DuckDBPyConnection,
        drafts: dict[str, dict[str, Any]],
    ) -> None:
        for draft_id, row in drafts.items():
            existing = con.execute(
                "SELECT expansion, event_type, draft_time FROM drafts WHERE draft_id = ?",
                [draft_id],
            ).fetchone()
            expected = (row["expansion"], row["event_type"], row["draft_time"])
            if existing is None:
                con.execute(
                    "INSERT INTO drafts (draft_id, expansion, event_type, draft_time) VALUES (?, ?, ?, ?)",
                    [draft_id, *expected],
                )
            elif not values_equal(existing, expected):
                raise ValueError(f"Draft metadata changed for {draft_id}")

    def build_deck_compositions(
        self,
        game_batch: pd.DataFrame,
    ) -> tuple[list[int | None], dict[tuple[str, int], dict[int, tuple[int, int]]]]:
        assert self.reference is not None
        items = [item for item in self.game_card_columns if item["stat"] in {"deck", "sideboard"}]
        build_columns = [item["column"] for item in items]
        if not build_columns:
            return [None] * len(game_batch), {}

        matrix = numeric_count_matrix(game_batch, build_columns).to_numpy(copy=False)
        card_ids = np.array(
            [self.reference.card_name_to_id[item["card_name"]] for item in items],
            dtype=np.int64,
        )
        slots = np.array([0 if item["stat"] == "deck" else 1 for item in items], dtype=np.int8)
        metadata = game_batch[["draft_id", "build_index"]].itertuples(index=False, name=None)
        compositions: dict[tuple[str, int], dict[int, tuple[int, int]]] = {}
        build_ids: list[int | None] = []

        for position, (draft_raw, build_raw) in enumerate(metadata):
            draft_id = str(draft_raw)
            build_index = integer_or_none(build_raw)
            if build_index is None:
                build_ids.append(None)
                continue
            counts_by_card: dict[int, list[int]] = {}
            for column_index in np.flatnonzero(matrix[position]):
                count = int(matrix[position, column_index])
                card_id = int(card_ids[column_index])
                slot = int(slots[column_index])
                counts_by_card.setdefault(card_id, [0, 0])[slot] = count
            composition = {card_id: (counts[0], counts[1]) for card_id, counts in counts_by_card.items()}
            key = (draft_id, build_index)
            previous = compositions.get(key)
            if previous is None:
                compositions[key] = composition
            elif previous != composition:
                LOG.warning(
                    "Deck composition changed within batch for draft_id=%s build_index=%s; "
                    "keeping first composition and ignoring duplicate.",
                    draft_id,
                    build_index,
                )
            build_ids.append(make_build_id(draft_id, build_index))
        return build_ids, compositions

    def ensure_deck_builds(
        self,
        con: duckdb.DuckDBPyConnection,
        compositions: dict[tuple[str, int], dict[int, tuple[int, int]]],
    ) -> None:
        for (draft_id, build_index), composition in compositions.items():
            build_id = make_build_id(draft_id, build_index)
            existing = con.execute(
                "SELECT build_id FROM deck_builds WHERE draft_id = ? AND build_index = ?",
                [draft_id, build_index],
            ).fetchone()
            if existing is None:
                con.execute(
                    "INSERT INTO deck_builds VALUES (?, ?, ?)",
                    [build_id, draft_id, build_index],
                )
                rows = [
                    {
                        "build_id": build_id,
                        "card_id": card_id,
                        "deck_count": counts[0],
                        "sideboard_count": counts[1],
                    }
                    for card_id, counts in composition.items()
                ]
                insert_rows(
                    con,
                    "deck_build_cards",
                    rows,
                    int64=("build_id", "card_id"),
                    int16=("deck_count", "sideboard_count"),
                )
                continue

            if int(existing[0]) != build_id:
                raise ValueError(
                    f"Deterministic build ID mismatch for {(draft_id, build_index)}"
                )

            stored = {
                int(card_id): (int(deck_count), int(sideboard_count))
                for card_id, deck_count, sideboard_count in con.execute(
                    """
                    SELECT card_id, deck_count, sideboard_count
                    FROM deck_build_cards
                    WHERE build_id = ?
                    """,
                    [build_id],
                ).fetchall()
            }

            if stored != composition:
                LOG.warning(
                    "Deck build already exists with different composition for "
                    "draft_id=%s build_index=%s; keeping existing database version.",
                    draft_id,
                    build_index,
                )

    def build_game_rows(
        self,
        game_batch: pd.DataFrame,
        game_ids: list[int],
        build_ids: list[int | None],
    ) -> list[dict[str, Any]]:
        columns = [
            column for column in (
                "draft_id", "match_number", "game_number", "game_time",
                "on_play", "won", "num_turns",
            )
            if column in game_batch.columns
        ]
        rows = []
        for position, record in enumerate(game_batch[columns].to_dict("records")):
            rows.append({
                "game_id": game_ids[position],
                "draft_id": str(record["draft_id"]),
                "build_id": build_ids[position],
                "match_number": integer_or_none(record.get("match_number")),
                "game_number": integer_or_none(record.get("game_number")),
                "game_time": timestamp_or_none(record.get("game_time")),
                "user_on_play": parse_bool(record.get("on_play")),
                "user_won": parse_bool(record.get("won")),
                "source_num_turns": integer_or_none(record.get("num_turns")),
            })
        return rows

    def build_player_rows(
        self,
        game_batch: pd.DataFrame,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> list[dict[str, Any]]:
        rows = []
        replay_records = replay_batch.to_dict("records")
        for position, game in enumerate(game_batch.to_dict("records")):
            replay = replay_records[position]
            game_id = game_ids[position]
            rows.append(
                {
                    "game_id": game_id,
                    "is_user": True,
                    "rank": text_or_none(game.get("rank")),
                    "main_colors": text_or_none(game.get("main_colors")),
                    "splash_colors": text_or_none(game.get("splash_colors")),
                    "observed_colors": None,
                    "num_mulligans": integer_or_none(game.get("num_mulligans")),
                    "n_games_bucket": integer_or_none(replay.get("user_n_games_bucket")),
                    "game_win_rate_bucket": numeric_or_none(replay.get("user_game_win_rate_bucket")),
                }
            )
            rows.append(
                {
                    "game_id": game_id,
                    "is_user": False,
                    "rank": text_or_none(game.get("opp_rank")),
                    "main_colors": None,
                    "splash_colors": None,
                    "observed_colors": text_or_none(game.get("opp_colors")),
                    "num_mulligans": integer_or_none(game.get("opp_num_mulligans")),
                    "n_games_bucket": None,
                    "game_win_rate_bucket": None,
                }
            )
        return rows

    def build_game_card_stats(
        self,
        game_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> list[dict[str, Any]]:
        assert self.reference is not None
        items = [item for item in self.game_card_columns if item["stat"] in GAME_STAT_PREFIXES]
        if not items:
            return []
        columns = [item["column"] for item in items]
        matrix = numeric_count_matrix(game_batch, columns).to_numpy(copy=False)
        card_ids = [self.reference.card_name_to_id[item["card_name"]] for item in items]
        stat_slots = [{"opening_hand": 0, "drawn": 1, "tutored": 2}[item["stat"]] for item in items]
        stats: dict[tuple[int, int], list[int]] = {}
        row_indices, column_indices = np.nonzero(matrix)
        for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist(), strict=True):
            count = int(matrix[row_index, column_index])
            if count < 0:
                raise ValueError(f"Negative card count in {columns[column_index]}")
            key = (game_ids[row_index], int(card_ids[column_index]))
            values = stats.setdefault(key, [0, 0, 0])
            values[stat_slots[column_index]] = count
        return [
            {
                "game_id": game_id, "card_id": card_id,
                "opening_hand_count": values[0], "drawn_count": values[1],
                "tutored_count": values[2],
            }
            for (game_id, card_id), values in stats.items()
        ]

    def turn_meaningful_positions(self, replay_batch: pd.DataFrame) -> dict[str, np.ndarray]:
        positions: dict[str, np.ndarray] = {}
        for column in self.turn_columns:
            if column not in replay_batch.columns:
                continue
            values = replay_batch[column].to_numpy(copy=False)
            # These frames are created directly from csv strings, so missing values are
            # empty strings. Avoid pandas StringDtype/strip/lower work in the hot path.
            mask = (values != "") & (values != "0") & (values != "0.0")
            found = np.flatnonzero(mask)
            if len(found):
                positions[column] = found
        return positions

    def find_turns(
        self,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
        positions_by_column: dict[str, np.ndarray] | None = None,
    ) -> set[tuple[int, str, int]]:
        active: set[tuple[int, str, int]] = set()
        positions_by_column = positions_by_column or self.turn_meaningful_positions(replay_batch)
        for column, positions in positions_by_column.items():
            source_side, source_turn_index, _ = self.turn_columns[column]
            for position in positions:
                active.add((game_ids[int(position)], source_side, source_turn_index))
        return active

    def make_turn_id_map(
        self,
        active: set[tuple[int, str, int]],
    ) -> dict[tuple[int, str, int], int]:
        # turn_id is reused by the turn row, events, states and zones. Hash it once.
        return {
            key: make_turn_id(*key)
            for key in active
        }

    def build_turn_rows(
        self,
        active: set[tuple[int, str, int]],
        turn_id_map: dict[tuple[int, str, int], int] | None = None,
    ) -> list[dict[str, Any]]:
        turn_id_map = turn_id_map or self.make_turn_id_map(active)
        return [
            {
                "turn_id": turn_id_map[(game_id, source_side, source_turn_index)],
                "game_id": game_id,
                "is_user_turn": source_side == "user",
                "source_turn_index": source_turn_index,
            }
            for game_id, source_side, source_turn_index in sorted(active)
        ]

    def build_event_rows(
        self,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
        positions_by_column: dict[str, np.ndarray] | None = None,
        turn_id_map: dict[tuple[int, str, int], int] | None = None,
    ) -> list[dict[str, Any]]:
        assert self.reference is not None
        rows: list[dict[str, Any]] = []
        positions_by_column = positions_by_column or self.turn_meaningful_positions(replay_batch)
        turn_id_map = turn_id_map or self.make_turn_id_map(
            self.find_turns(replay_batch, game_ids, positions_by_column)
        )
        for column, rule in self.event_columns.items():
            positions = positions_by_column.get(column)
            if positions is None:
                continue
            values = replay_batch[column].to_numpy(copy=False)
            source_side, source_turn_index, _ = self.turn_columns[column]
            actor = resolve_player(rule.actor, source_side)
            affected = resolve_player(rule.affected, source_side)
            for position_raw in positions:
                position = int(position_raw)
                game_id = game_ids[position]
                value = values[position]
                source_turn_id = turn_id_map[(game_id, source_side, source_turn_index)]
                actual_turn_id = source_turn_id if rule.exact_turn else None
                if rule.payload == "card":
                    for ordinal, token in enumerate(split_ids(value), start=1):
                        arena_id = parse_source_id(token)
                        rows.append(event_row(
                            game_id=game_id, source_turn_id=source_turn_id, actual_turn_id=actual_turn_id,
                            source_field=column, source_ordinal=ordinal, event_type=rule.event_type,
                            actor_is_user=actor, affected_is_user=affected, source_arena_card_id=arena_id,
                            card_id=self.reference.arena_to_card_id.get(arena_id),
                        ))
                elif rule.payload == "ability":
                    for ordinal, token in enumerate(split_ids(value), start=1):
                        ability_id = parse_source_id(token)
                        rows.append(event_row(
                            game_id=game_id, source_turn_id=source_turn_id, actual_turn_id=actual_turn_id,
                            source_field=column, source_ordinal=ordinal, event_type=rule.event_type,
                            actor_is_user=actor, affected_is_user=affected, source_ability_id=ability_id,
                            ability_id=ability_id if ability_id in self.reference.ability_ids else None,
                        ))
                elif rule.payload == "numeric":
                    numeric = numeric_or_none(value)
                    if numeric is not None and numeric != 0:
                        rows.append(event_row(
                            game_id=game_id, source_turn_id=source_turn_id, actual_turn_id=actual_turn_id,
                            source_field=column, source_ordinal=1, event_type=rule.event_type,
                            actor_is_user=actor, affected_is_user=affected, numeric_value=numeric,
                        ))
                else:
                    raise ValueError(f"Unknown event payload {rule.payload!r}")
        return rows

    def build_candidate_hand_rows(
            self,
            replay_batch: pd.DataFrame,
            game_ids: list[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert self.reference is not None

        hand_rows: list[dict[str, Any]] = []
        card_rows: list[dict[str, Any]] = []

        columns = ["num_mulligans", "opening_hand", *self.candidate_columns]
        arrays = {
            column: replay_batch[column].to_numpy(copy=False)
            for column in columns
            if column in replay_batch.columns
        }

        for position, game_id in enumerate(game_ids):
            mulligans = integer_or_none(arrays["num_mulligans"][position])
            opening_cards = split_ids(arrays["opening_hand"][position])

            candidates: list[tuple[int, list[str]]] = []

            for column in self.candidate_columns:
                attempt = int(column.rsplit("_", 1)[1])
                cards = split_ids(arrays[column][position])

                if not cards:
                    continue

                if len(cards) != 7:
                    LOG.warning(
                        "Source inconsistency: candidate hand %s for game %s "
                        "contains %s cards instead of 7; skipping that candidate.",
                        attempt,
                        game_id,
                        len(cards),
                    )
                    continue

                candidates.append((attempt, cards))

            if not candidates:
                raise ValueError(
                    f"Replay game {game_id} has no usable seven-card candidate hands"
                )

            expected_candidates = mulligans + 1 if mulligans is not None else None

            if expected_candidates is None or len(candidates) != expected_candidates:
                LOG.warning(
                    "Source inconsistency: game %s reports num_mulligans=%r but "
                    "contains %s usable candidate hands; preserving observed candidates.",
                    game_id,
                    mulligans,
                    len(candidates),
                )

            # Prefer the candidate that exactly matches opening_hand. This identifies
            # the kept seven-card candidate even when num_mulligans disagrees with
            # the populated candidate_hand_N fields.
            matching_attempts = [
                attempt
                for attempt, cards in candidates
                if Counter(cards) == Counter(opening_cards)
            ]

            if matching_attempts:
                # The same seven-card composition could theoretically appear more
                # than once, so the latest matching source attempt is the final one.
                final_attempt = matching_attempts[-1]
            else:
                # Preserve the observed candidate sequence rather than failing the
                # entire ingestion because opening_hand disagrees with it.
                final_attempt = candidates[-1][0]

                LOG.warning(
                    "Source inconsistency: opening_hand does not match any candidate "
                    "hand for game %s; treating the last observed candidate "
                    "(attempt %s) as final.",
                    game_id,
                    final_attempt,
                )

            for attempt, cards in candidates:
                hand_id = make_hand_id(game_id, attempt)

                hand_rows.append({
                    "hand_id": hand_id,
                    "game_id": game_id,
                    "attempt_number": attempt,
                    "is_final_candidate": attempt == final_attempt,
                })

                for slot_number, token in enumerate(cards, start=1):
                    arena_id = parse_source_id(token)

                    card_rows.append({
                        "hand_id": hand_id,
                        "slot_number": slot_number,
                        "source_arena_card_id": arena_id,
                        "card_id": self.reference.arena_to_card_id.get(arena_id),
                    })

        return hand_rows, card_rows

    def build_turn_state_and_zones(
        self,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
        active_turns: set[tuple[int, str, int]],
        turn_id_map: dict[tuple[int, str, int], int] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert self.reference is not None
        positions = {game_id: position for position, game_id in enumerate(game_ids)}
        turn_id_map = turn_id_map or self.make_turn_id_map(active_turns)
        arrays = {column: replay_batch[column].to_numpy(copy=False) for column in replay_batch.columns}
        state_rows: list[dict[str, Any]] = []
        zone_rows: list[dict[str, Any]] = []

        def raw(column: str, position: int) -> Any:
            values = arrays.get(column)
            return None if values is None else values[position]

        for game_id, source_side, source_turn_index in sorted(active_turns):
            position = positions[game_id]
            turn_id = turn_id_map[(game_id, source_side, source_turn_index)]
            stem = f"{source_side}_turn_{source_turn_index}_"
            for player_is_user, player in ((True, "user"), (False, "oppo")):
                life = numeric_or_none(raw(f"{stem}eot_{player}_life", position))
                poison = numeric_or_none(raw(f"{stem}eot_{player}_poison_counters", position))
                mana_spent = numeric_or_none(raw(f"{stem}{player}_mana_spent", position))
                if player_is_user:
                    hand_raw = raw(f"{stem}eot_user_cards_in_hand", position)
                    hand_size = None if is_missing_or_blank(hand_raw) else len(split_ids(hand_raw))
                else:
                    hand_size = integer_or_none(raw(f"{stem}eot_oppo_cards_in_hand", position))
                if any(value is not None for value in (life, poison, mana_spent, hand_size)):
                    state_rows.append({
                        "turn_id": turn_id, "player_is_user": player_is_user, "life": life,
                        "poison_counters": poison, "hand_size": hand_size, "mana_spent": mana_spent,
                    })
            for suffix, (owner_is_user, zone) in ZONE_FIELDS.items():
                counts = Counter(parse_source_id(token) for token in split_ids(raw(stem + suffix, position)))
                for arena_id, quantity in counts.items():
                    zone_rows.append({
                        "turn_id": turn_id, "owner_is_user": owner_is_user, "zone": zone,
                        "source_arena_card_id": arena_id,
                        "card_id": self.reference.arena_to_card_id.get(arena_id), "quantity": quantity,
                    })
        return state_rows, zone_rows

    def build_total_rows(
        self,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> list[dict[str, Any]]:
        arrays = {column: replay_batch[column].to_numpy(copy=False) for column in replay_batch.columns}
        rows = []
        for position, game_id in enumerate(game_ids):
            for is_user, side in ((True, "user"), (False, "oppo")):
                row: dict[str, Any] = {"game_id": game_id, "is_user": is_user}
                for output_name, source_suffix in TOTAL_FIELDS.items():
                    source_column = f"{side}_{source_suffix}"
                    values = arrays.get(source_column)
                    row[output_name] = None if values is None else numeric_or_none(values[position])
                rows.append(row)
        return rows

    def mark_complete(
        self,
        con: duckdb.DuckDBPyConnection,
        checkpoint: Checkpoint,
    ) -> None:
        con.execute("BEGIN")
        try:
            con.execute(
                """
                UPDATE ingestion_checkpoints
                SET completed = TRUE,
                    next_draft_index = ?,
                    committed_drafts = ?,
                    updated_at = current_timestamp
                WHERE dataset_id = ?
                """,
                [checkpoint.next_draft_index, checkpoint.committed_drafts, self.paths.dataset_id],
            )
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("CHECKPOINT")

    def run_integrity_checks(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        emit_info: bool = True,
    ) -> None:
        checks = {
            "game-backed drafts without draft picks": """
                SELECT count(*) FROM (
                    SELECT DISTINCT g.draft_id
                    FROM games g
                    LEFT JOIN draft_picks p ON p.draft_id = g.draft_id
                    WHERE p.draft_id IS NULL
                )
            """,
            "selected draft cards absent from offered pack": """
                SELECT count(*)
                FROM draft_picks p
                LEFT JOIN draft_pick_cards c
                  ON c.pick_id = p.pick_id
                 AND c.card_id = p.card_id
                 AND c.pack_count > 0
                WHERE c.card_id IS NULL
            """,
            "negative draft pack/pool counts": """
                SELECT count(*)
                FROM draft_pick_cards
                WHERE pack_count < 0 OR pool_count < 0
            """,
            "games without exactly two player rows": """
                SELECT count(*) FROM (
                    SELECT g.game_id
                    FROM games g
                    LEFT JOIN game_players gp USING (game_id)
                    GROUP BY g.game_id
                    HAVING count(gp.is_user) <> 2
                )
            """,
            "games without exactly one final candidate": """
                SELECT count(*) FROM (
                    SELECT g.game_id
                    FROM games g
                    LEFT JOIN candidate_hands h USING (game_id)
                    GROUP BY g.game_id
                    HAVING coalesce(sum(CASE WHEN h.is_final_candidate THEN 1 ELSE 0 END), 0) <> 1
                )
            """,
            "events whose source turn belongs to another game": """
                SELECT count(*)
                FROM events e
                JOIN turns t ON t.turn_id = e.source_turn_id
                WHERE t.game_id <> e.game_id
            """,
            "events whose actual turn belongs to another game": """
                SELECT count(*)
                FROM events e
                JOIN turns t ON t.turn_id = e.actual_turn_id
                WHERE e.actual_turn_id IS NOT NULL AND t.game_id <> e.game_id
            """,
        }
        for label, sql in checks.items():
            failures = int(con.execute(sql).fetchone()[0])
            if failures:
                raise ValueError(f"Integrity check failed: {label}: {failures}")
            if emit_info:
                LOG.info("PASS: %s", label)

        unmapped_cards = int(con.execute(
            """
            SELECT count(DISTINCT source_arena_card_id) FROM (
                SELECT source_arena_card_id, card_id FROM events
                UNION ALL
                SELECT source_arena_card_id, card_id FROM candidate_hand_cards
                UNION ALL
                SELECT source_arena_card_id, card_id FROM turn_zone_cards
            ) refs
            WHERE source_arena_card_id > 0 AND card_id IS NULL
            """
        ).fetchone()[0])
        unmapped_abilities = int(con.execute(
            """
            SELECT count(DISTINCT source_ability_id)
            FROM events
            WHERE source_ability_id > 0 AND ability_id IS NULL
            """
        ).fetchone()[0])
        unknown_turn_events = int(con.execute(
            "SELECT count(*) FROM events WHERE actual_turn_id IS NULL"
        ).fetchone()[0])
        total_games = int(con.execute("SELECT count(*) FROM games").fetchone()[0])
        total_events = int(con.execute("SELECT count(*) FROM events").fetchone()[0])

        if emit_info:
            LOG.info("Database rows: %d games; %d events", total_games, total_events)
            LOG.info("Events with intentionally unknown actual turn: %d", unknown_turn_events)
        if unmapped_cards:
            LOG.warning("Positive replay Arena IDs without canonical mapping: %d", unmapped_cards)
        if unmapped_abilities:
            LOG.warning("Positive replay ability IDs without canonical mapping: %d", unmapped_abilities)


def classify_event_suffix(source_side: str, suffix: str) -> EventRule | None:
    direct = {
        "cards_discarded": EventRule("card_discarded", "card", "source", exact_turn=False),
        "lands_played": EventRule("land_played", "card", "source", exact_turn=True),
        "creatures_cast": EventRule("creature_cast", "card", "source", exact_turn=False),
        "non_creatures_cast": EventRule("non_creature_cast", "card", "source", exact_turn=False),
        "creatures_attacked": EventRule("creature_attacked", "card", "source", exact_turn=True),
        "creatures_blocked": EventRule("creature_blocked", "card", "source", exact_turn=True),
        "creatures_unblocked": EventRule("creature_unblocked", "card", "source", exact_turn=True),
        "creatures_blocking": EventRule("creature_blocking", "card", "opposite_source", exact_turn=True),
    }
    if suffix in direct:
        return direct[suffix]

    if source_side == "user" and suffix in {"cards_drawn", "cards_tutored", "cards_foretold"}:
        return EventRule(
            {
                "cards_drawn": "card_drawn",
                "cards_tutored": "card_tutored",
                "cards_foretold": "card_foretold",
            }[suffix],
            "card",
            "user",
            exact_turn=suffix == "cards_foretold",
        )

    if source_side == "oppo" and suffix in {
        "cards_drawn",
        "cards_tutored",
        "cards_drawn_or_tutored",
        "cards_foretold",
    }:
        return EventRule(
            {
                "cards_drawn": "cards_drawn_hidden",
                "cards_tutored": "cards_tutored_hidden",
                "cards_drawn_or_tutored": "cards_drawn_or_tutored_hidden",
                "cards_foretold": "cards_foretold_hidden",
            }[suffix],
            "numeric",
            "oppo",
            exact_turn=suffix == "cards_foretold",
        )

    explicit = re.fullmatch(r"(user|oppo)_(instants_sorceries_cast|cards_learned|abilities)", suffix)
    if explicit:
        actor, family = explicit.groups()
        return EventRule(
            {
                "instants_sorceries_cast": "instant_sorcery_cast",
                "cards_learned": "card_learned",
                "abilities": "ability",
            }[family],
            "ability" if family == "abilities" else "card",
            actor,
            exact_turn=True,
        )

    death = re.fullmatch(r"(user|oppo)_creatures_killed_(combat|non_combat)", suffix)
    if death:
        affected, death_type = death.groups()
        return EventRule(f"creature_died_{death_type}", "card", "unknown", affected, True)

    damage = re.fullmatch(r"(user|oppo)_combat_damage_taken", suffix)
    if damage:
        affected = damage.group(1)
        return EventRule("combat_damage_taken", "numeric", opposite_side(affected), affected, True)

    if suffix == "creatures_blitzed":
        return EventRule("creatures_blitzed", "numeric", "source", exact_turn=False)

    if suffix == "player_combat_damage_dealt":
        return EventRule("combat_damage_dealt", "numeric", "source", "opposite_source", True)

    return None


def is_state_or_zone_suffix(suffix: str) -> bool:
    if suffix in ZONE_FIELDS:
        return True
    if re.fullmatch(r"eot_(user|oppo)_(life|poison_counters|cards_in_hand)", suffix):
        return True
    if re.fullmatch(r"(user|oppo)_mana_spent", suffix):
        return True
    return False


def validate_turn_schema(turn_columns: dict[str, tuple[str, int, str]]) -> None:
    unknown = sorted(
        {
            suffix
            for source_side, _, suffix in turn_columns.values()
            if classify_event_suffix(source_side, suffix) is None and not is_state_or_zone_suffix(suffix)
        }
    )
    if unknown:
        raise ValueError(f"Unclassified replay turn suffixes: {unknown}")


def validate_repeated_turn_schema(turn_columns: dict[str, tuple[str, int, str]]) -> None:
    by_side: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for source_side, source_turn_index, suffix in turn_columns.values():
        by_side[source_side][source_turn_index].add(suffix)
    for source_side, turns in by_side.items():
        if not turns:
            continue
        baseline_index = min(turns)
        baseline = turns[baseline_index]
        for index, suffixes in turns.items():
            if suffixes != baseline:
                raise ValueError(
                    f"Replay {source_side}_turn_{index} schema differs from "
                    f"{source_side}_turn_{baseline_index}"
                )


def validate_required_columns(
    game_columns: list[str],
    replay_columns: list[str],
    candidate_columns: list[str],
) -> None:
    require_columns(game_columns, GAME_METADATA_COLUMNS, "game CSV")
    require_columns(replay_columns, {*GAME_KEY, "num_mulligans", "opening_hand"}, "replay CSV")
    if not candidate_columns:
        raise ValueError("Replay CSV has no candidate_hand_N columns")
    attempts = [int(column.rsplit("_", 1)[1]) for column in candidate_columns]
    if attempts != list(range(1, max(attempts) + 1)):
        raise ValueError(f"Candidate hand columns are not contiguous: {attempts}")


def validate_database_schema(con: duckdb.DuckDBPyConnection) -> None:
    required = {
        "drafts": {
            "draft_id", "expansion", "event_type", "draft_time", "rank",
            "event_match_wins", "event_match_losses", "user_n_games_bucket",
            "user_game_win_rate_bucket",
        },
        "draft_picks": {
            "pick_id", "draft_id", "pack_number", "pick_number", "card_id",
            "pick_maindeck_rate", "pick_sideboard_in_rate",
        },
        "draft_pick_cards": {"pick_id", "card_id", "pack_count", "pool_count"},
        "games": {"game_id", "draft_id", "build_id", "source_num_turns"},
        "turns": {"turn_id", "game_id", "is_user_turn", "source_turn_index"},
        "events": {"event_id", "source_turn_id", "actual_turn_id", "actor_is_user"},
        "ingestion_checkpoints": {"dataset_id", "next_draft_index", "committed_drafts", "completed"},
    }
    for table, columns in required.items():
        existing = {row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
        missing = columns - existing
        if missing:
            raise ValueError(
                f"Existing database has an incompatible {table} table; missing {sorted(missing)}. "
                "Rebuild with --overwrite."
            )

def validate_game_replay_pair(game_batch: pd.DataFrame, replay_batch: pd.DataFrame) -> None:
    if game_keys(game_batch) != game_keys(replay_batch):
        raise ValueError("Game/replay natural keys do not align")
    for field, kind in PAIR_CHECK_FIELDS.items():
        if field not in game_batch.columns or field not in replay_batch.columns:
            continue
        for position, (game_value, replay_value) in enumerate(
            zip(game_batch[field], replay_batch[field], strict=True)
        ):
            if kind == "bool":
                left = parse_bool(game_value)
                right = parse_bool(replay_value)
            elif kind == "int":
                left = integer_or_none(game_value)
                right = integer_or_none(replay_value)
            else:
                raise ValueError(kind)
            if left != right:
                raise ValueError(
                    f"Game/replay mismatch for {field} at batch row {position}: {left!r} != {right!r}"
                )


def validate_batch_rows(
    game_ids: list[int],
    turn_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    hand_rows: list[dict[str, Any]],
    hand_card_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    zone_rows: list[dict[str, Any]],
) -> None:
    game_id_set = set(game_ids)
    turn_ids = {row["turn_id"] for row in turn_rows}
    hand_ids = {row["hand_id"] for row in hand_rows}
    if len(turn_ids) != len(turn_rows):
        raise ValueError("Duplicate turn_id generated inside batch")
    if len(hand_ids) != len(hand_rows):
        raise ValueError("Duplicate hand_id generated inside batch")
    if any(row["game_id"] not in game_id_set for row in turn_rows):
        raise ValueError("Turn references game outside current batch")
    if any(row["source_turn_id"] not in turn_ids for row in event_rows):
        raise ValueError("Event source_turn_id was not generated in current batch")
    if any(
        row["actual_turn_id"] is not None and row["actual_turn_id"] not in turn_ids
        for row in event_rows
    ):
        raise ValueError("Event actual_turn_id was not generated in current batch")
    if any(row["hand_id"] not in hand_ids for row in hand_card_rows):
        raise ValueError("Candidate hand card references unknown hand")
    if any(row["turn_id"] not in turn_ids for row in state_rows):
        raise ValueError("Turn state references unknown turn")
    if any(row["turn_id"] not in turn_ids for row in zone_rows):
        raise ValueError("Turn zone row references unknown turn")

    final_counts = Counter(row["game_id"] for row in hand_rows if row["is_final_candidate"])
    if any(final_counts[game_id] != 1 for game_id in game_ids):
        raise ValueError("Every game must have exactly one final candidate hand")


def parse_game_card_columns(columns: Iterable[str]) -> list[dict[str, str]]:
    parsed = []
    for column in columns:
        for prefix in GAME_CARD_PREFIXES:
            marker = prefix + "_"
            if column.startswith(marker):
                parsed.append(
                    {
                        "column": column,
                        "stat": prefix,
                        "card_name": normalize_card_name(column[len(marker):]),
                    }
                )
                break
    return parsed


def parse_turn_columns(columns: Iterable[str]) -> dict[str, tuple[str, int, str]]:
    result = {}
    for column in columns:
        match = TURN_RE.fullmatch(column)
        if match:
            result[column] = (match.group(1), int(match.group(2)), match.group(3))
    return result


def build_reference_data(
    helper_cards: pd.DataFrame,
    helper_abilities: pd.DataFrame,
    game_card_columns: list[dict[str, str]],
    draft_card_names: Iterable[str] = (),
) -> ReferenceData:
    helper_names = {
        normalize_card_name(name)
        for name in helper_cards["name"].dropna().astype(str)
        if normalize_card_name(name)
    }
    game_names = {item["card_name"] for item in game_card_columns}
    draft_names = {normalize_card_name(name) for name in draft_card_names if normalize_card_name(name)}
    names = sorted(helper_names | game_names | draft_names)
    card_name_to_id = {name: make_card_id(name) for name in names}

    arena_to_card_id = {}
    for arena_id_raw, name_raw in helper_cards[["id", "name"]].itertuples(index=False, name=None):
        arena_id = integer_or_none(arena_id_raw)
        name = text_or_none(name_raw)
        if arena_id is not None and name is not None:
            arena_to_card_id[arena_id] = card_name_to_id[normalize_card_name(name)]

    ability_ids = {
        ability_id
        for raw in helper_abilities["id"]
        if (ability_id := integer_or_none(raw)) is not None
    }
    return ReferenceData(card_name_to_id, arena_to_card_id, ability_ids)


def event_row(
    *,
    game_id: int,
    source_turn_id: int,
    actual_turn_id: int | None,
    source_field: str,
    source_ordinal: int,
    event_type: str,
    actor_is_user: bool | None,
    affected_is_user: bool | None,
    source_arena_card_id: int | None = None,
    card_id: int | None = None,
    source_ability_id: int | None = None,
    ability_id: int | None = None,
    numeric_value: float | None = None,
) -> dict[str, Any]:
    return {
        "event_id": make_event_id(game_id, source_field, source_ordinal),
        "game_id": game_id,
        "source_turn_id": source_turn_id,
        "actual_turn_id": actual_turn_id,
        "event_type": event_type,
        "actor_is_user": actor_is_user,
        "affected_is_user": affected_is_user,
        "source_arena_card_id": source_arena_card_id,
        "card_id": card_id,
        "source_ability_id": source_ability_id,
        "ability_id": ability_id,
        "numeric_value": numeric_value,
        "source_ordinal": source_ordinal,
        "source_field": source_field,
    }


def resolve_player(rule: str | None, source_side: str) -> bool | None:
    if rule is None or rule == "unknown":
        return None
    if rule == "source":
        return source_side == "user"
    if rule == "opposite_source":
        return source_side == "oppo"
    if rule == "user":
        return True
    if rule == "oppo":
        return False
    raise ValueError(f"Unknown player rule {rule!r}")


def opposite_side(side: str) -> str:
    return "oppo" if side == "user" else "user"


def deterministic_bigint(*parts: Any) -> int:
    value = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def make_game_id_from_parts(draft_id: Any, match_number: Any, game_number: Any) -> int:
    match = integer_or_none(match_number)
    game = integer_or_none(game_number)
    if is_missing_or_blank(draft_id) or match is None or game is None:
        raise ValueError(f"Invalid game key: {(draft_id, match_number, game_number)!r}")
    return deterministic_bigint(str(draft_id), match, game)


def make_game_ids(frame: pd.DataFrame) -> list[int]:
    return [
        make_game_id_from_parts(draft_id, match_number, game_number)
        for draft_id, match_number, game_number in frame[list(GAME_KEY)].itertuples(index=False, name=None)
    ]


def game_keys(frame: pd.DataFrame) -> list[tuple[str, int, int]]:
    result = []
    for draft_id, match_number, game_number in frame[list(GAME_KEY)].itertuples(index=False, name=None):
        match = integer_or_none(match_number)
        game = integer_or_none(game_number)
        if is_missing_or_blank(draft_id) or match is None or game is None:
            raise ValueError(f"Invalid game key: {(draft_id, match_number, game_number)!r}")
        result.append((str(draft_id), match, game))
    return result


def make_card_id(card_name: str) -> int:
    return deterministic_bigint("card", normalize_card_name(card_name))


def make_build_id(draft_id: str, build_index: int) -> int:
    return deterministic_bigint("build", draft_id, build_index)


def make_hand_id(game_id: int, attempt_number: int) -> int:
    return deterministic_bigint("hand", game_id, attempt_number)


def make_turn_id(game_id: int, source_side: str, source_turn_index: int) -> int:
    return deterministic_bigint("turn", game_id, source_side, source_turn_index)


def make_event_id(game_id: int, source_field: str, source_ordinal: int) -> int:
    return deterministic_bigint("event", game_id, source_field, source_ordinal)


def split_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value == "":
            return []
        # Replay ID lists are normally already canonical integer strings separated
        # by pipes. Avoid strip/lower/list filtering on that common path.
        if " " not in value and "\t" not in value and value not in {"nan", "none", "<na>"}:
            return value.split("|")
        text = value.strip()
    else:
        if is_missing_or_blank(value):
            return []
        text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    return [token.strip() for token in text.split("|") if token.strip()]


def parse_source_id(token: Any) -> int:
    if isinstance(token, str):
        try:
            return int(token)
        except ValueError:
            text = token.strip()
            return int(float(text))
    try:
        return int(token)
    except (TypeError, ValueError):
        return int(float(token))


def parse_key_integer(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return int(float(value))


def parse_bool(value: Any) -> bool | None:
    if is_missing_or_blank(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def integer_or_none(value: Any) -> int | None:
    numeric = numeric_or_none(value)
    if numeric is None:
        return None
    if not float(numeric).is_integer():
        raise ValueError(f"Expected integer, got {value!r}")
    return int(numeric)


def numeric_or_none(value: Any) -> float | None:
    if is_missing_or_blank(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(numeric) else numeric


def text_or_none(value: Any) -> str | None:
    if is_missing_or_blank(value):
        return None
    text = str(value).strip()
    return text or None


def timestamp_or_none(value: Any) -> pd.Timestamp | None:
    if is_missing_or_blank(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed


def is_missing_or_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def meaningful_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return series.notna() & ~text.isin(["", "0", "0.0", "nan", "none", "<na>"])


def normalize_card_name(name: str) -> str:
    return str(name).strip()


def numeric_count_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    raw = frame[columns].to_numpy(copy=False)
    try:
        # Source frames come directly from csv.reader, so blank and integer strings
        # dominate. NumPy conversion avoids one pd.to_numeric call per wide column.
        normalized = np.where(raw == "", "0", raw)
        matrix = normalized.astype(np.int16)
        if (matrix < 0).any():
            raise ValueError("Negative deck/sideboard count found")
        return pd.DataFrame(matrix, columns=columns, index=frame.index)
    except (TypeError, ValueError):
        matrix = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        if (matrix < 0).any().any():
            raise ValueError("Negative deck/sideboard count found")
        return matrix.astype("int16")


def clean_json_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        if is_missing_or_blank(value):
            cleaned[key] = None
        elif isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def values_equal(left: Iterable[Any], right: Iterable[Any]) -> bool:
    for a, b in zip(left, right, strict=True):
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if isinstance(a, pd.Timestamp) or isinstance(b, pd.Timestamp):
            if pd.Timestamp(a) != pd.Timestamp(b):
                return False
        elif isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            # DuckDB REAL is IEEE float32, so a source value such as 0.48 is
            # returned as 0.47999998927116394. Treat normal float32 roundoff
            # as equal while still rejecting materially different values.
            if not np.isclose(float(a), float(b), rtol=1e-6, atol=1e-7):
                return False
        elif a != b:
            return False
    return True


def rows_to_frame(
    rows: list[dict[str, Any]],
    *,
    int64: Iterable[str] = (),
    int16: Iterable[str] = (),
    boolean: Iterable[str] = (),
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    # Nullable 63-bit IDs must be built from original ints; float64 would round them.
    for column in int64:
        if column in frame.columns:
            frame[column] = pd.array([row.get(column) for row in rows], dtype="Int64")
    for column in int16:
        if column in frame.columns:
            frame[column] = pd.array([row.get(column) for row in rows], dtype="Int16")
    for column in boolean:
        if column in frame.columns:
            frame[column] = pd.array([row.get(column) for row in rows], dtype="boolean")
    return frame


def insert_rows(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[dict[str, Any]],
    *,
    int64: Iterable[str] = (),
    int16: Iterable[str] = (),
    boolean: Iterable[str] = (),
) -> None:
    if not rows:
        return
    frame = rows_to_frame(rows, int64=int64, int16=int16, boolean=boolean)
    temp_name = f"_insert_{table_name}"
    con.register(temp_name, frame)
    try:
        columns = list(frame.columns)
        quoted = ", ".join(f'"{column}"' for column in columns)
        con.execute(f"INSERT INTO {table_name} ({quoted}) SELECT {quoted} FROM {temp_name}")
    finally:
        con.unregister(temp_name)


def insert_or_validate_draft_rows(
    con: duckdb.DuckDBPyConnection,
    pick_rows: list[tuple[int, str, int, int, int, float | None, float | None]],
    pick_card_rows: list[tuple[int, int, int, int]],
    *,
    validate_existing: bool = True,
) -> None:
    """Insert missing draft rows and verify any rows already present match source.

    Draft ingestion uses tuples rather than one Python dict per normalized row. This
    matters for draft_pick_cards, where a 10k-pick chunk can contain hundreds of
    thousands of child rows.
    """
    if pick_rows:
        pick_frame = pd.DataFrame.from_records(
            pick_rows,
            columns=(
                "pick_id", "draft_id", "pack_number", "pick_number", "card_id",
                "pick_maindeck_rate", "pick_sideboard_in_rate",
            ),
        )
        pick_frame["pick_id"] = pick_frame["pick_id"].astype("int64")
        pick_frame["card_id"] = pick_frame["card_id"].astype("int64")
        pick_frame["pack_number"] = pick_frame["pack_number"].astype("int16")
        pick_frame["pick_number"] = pick_frame["pick_number"].astype("int16")
        con.register("_draft_pick_chunk", pick_frame)
        try:
            if validate_existing:
                mismatch = con.execute(
                    """
                    SELECT s.pick_id
                    FROM _draft_pick_chunk s
                    JOIN draft_picks p USING (pick_id)
                    WHERE p.draft_id IS DISTINCT FROM s.draft_id
                       OR p.pack_number IS DISTINCT FROM s.pack_number
                       OR p.pick_number IS DISTINCT FROM s.pick_number
                       OR p.card_id IS DISTINCT FROM s.card_id
                       OR CAST(p.pick_maindeck_rate AS REAL)
                          IS DISTINCT FROM CAST(s.pick_maindeck_rate AS REAL)
                       OR CAST(p.pick_sideboard_in_rate AS REAL)
                          IS DISTINCT FROM CAST(s.pick_sideboard_in_rate AS REAL)
                    LIMIT 10
                    """
                ).fetchall()
                if mismatch:
                    raise ValueError(
                        f"Existing draft_picks rows differ from source for pick IDs: "
                        f"{[int(row[0]) for row in mismatch]}"
                    )
                con.execute(
                    """
                    INSERT INTO draft_picks (
                        pick_id, draft_id, pack_number, pick_number, card_id,
                        pick_maindeck_rate, pick_sideboard_in_rate
                    )
                    SELECT
                        s.pick_id, s.draft_id, s.pack_number, s.pick_number, s.card_id,
                        s.pick_maindeck_rate, s.pick_sideboard_in_rate
                    FROM _draft_pick_chunk s
                    LEFT JOIN draft_picks p USING (pick_id)
                    WHERE p.pick_id IS NULL
                    """
                )
            else:
                con.execute(
                    """
                    INSERT INTO draft_picks (
                        pick_id, draft_id, pack_number, pick_number, card_id,
                        pick_maindeck_rate, pick_sideboard_in_rate
                    )
                    SELECT pick_id, draft_id, pack_number, pick_number, card_id,
                           pick_maindeck_rate, pick_sideboard_in_rate
                    FROM _draft_pick_chunk
                    """
                )
        finally:
            con.unregister("_draft_pick_chunk")

    if pick_card_rows:
        card_frame = pd.DataFrame.from_records(
            pick_card_rows,
            columns=("pick_id", "card_id", "pack_count", "pool_count"),
        )
        card_frame["pick_id"] = card_frame["pick_id"].astype("int64")
        card_frame["card_id"] = card_frame["card_id"].astype("int64")
        card_frame["pack_count"] = card_frame["pack_count"].astype("int16")
        card_frame["pool_count"] = card_frame["pool_count"].astype("int16")
        con.register("_draft_pick_card_chunk", card_frame)
        try:
            if validate_existing:
                mismatch = con.execute(
                    """
                    SELECT s.pick_id, s.card_id
                    FROM _draft_pick_card_chunk s
                    JOIN draft_pick_cards c
                      ON c.pick_id = s.pick_id AND c.card_id = s.card_id
                    WHERE c.pack_count IS DISTINCT FROM s.pack_count
                       OR c.pool_count IS DISTINCT FROM s.pool_count
                    LIMIT 10
                    """
                ).fetchall()
                if mismatch:
                    raise ValueError(
                        "Existing draft_pick_cards rows differ from source for keys: "
                        f"{[(int(a), int(b)) for a, b in mismatch]}"
                    )
                con.execute(
                    """
                    INSERT INTO draft_pick_cards (
                        pick_id, card_id, pack_count, pool_count
                    )
                    SELECT s.pick_id, s.card_id, s.pack_count, s.pool_count
                    FROM _draft_pick_card_chunk s
                    LEFT JOIN draft_pick_cards c
                      ON c.pick_id = s.pick_id AND c.card_id = s.card_id
                    WHERE c.pick_id IS NULL
                    """
                )
            else:
                con.execute(
                    """
                    INSERT INTO draft_pick_cards (pick_id, card_id, pack_count, pool_count)
                    SELECT pick_id, card_id, pack_count, pool_count
                    FROM _draft_pick_card_chunk
                    """
                )
        finally:
            con.unregister("_draft_pick_card_chunk")


def open_csv_text(path: Path):
    if path.suffix == ".gz":
        raw = path.open("rb", buffering=8 * 1024 * 1024)
        gz = gzip.GzipFile(fileobj=raw, mode="rb")
        buffered = io.BufferedReader(gz, buffer_size=8 * 1024 * 1024)
        text = io.TextIOWrapper(buffered, encoding="utf-8", newline="")
        text._compressed_raw = raw
        text._compressed_size = path.stat().st_size
        return text
    text = path.open("r", encoding="utf-8", newline="", buffering=8 * 1024 * 1024)
    text._compressed_raw = text.buffer
    text._compressed_size = path.stat().st_size
    return text


def read_header(path: Path) -> list[str]:
    with open_csv_text(path) as handle:
        return next(csv.reader(handle))


def file_signature(path: Path, header: list[str], sample_bytes: int = 1_048_576) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    digest.update("\x1f".join(header).encode("utf-8"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def require_columns(columns: Iterable[str], required: set[str], source_name: str) -> None:
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {sorted(missing)}")


def prepare_output(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path.unlink(missing_ok=True)
        Path(str(path) + ".wal").unlink(missing_ok=True)


def derive_dataset_id(game_file: Path, replay_file: Path, requested: str | None) -> str:
    if requested:
        return requested
    game_prefix = "game_data_public."
    replay_prefix = "replay_data_public."
    suffix = ".csv.gz"
    if (
        game_file.name.startswith(game_prefix)
        and replay_file.name.startswith(replay_prefix)
        and game_file.name.endswith(suffix)
        and replay_file.name.endswith(suffix)
    ):
        game_id = game_file.name[len(game_prefix):-len(suffix)]
        replay_id = replay_file.name[len(replay_prefix):-len(suffix)]
        if game_id == replay_id:
            return game_id
    digest = hashlib.sha256(f"{game_file.name}|{replay_file.name}".encode("utf-8")).hexdigest()[:16]
    return f"manual-{digest}"


def discover_dataset_pairs(
    data_dir: Path,
    suffix: str | None,
) -> list[tuple[str, Path, Path, Path]]:
    draft_files = sorted(data_dir.glob("draft_data_public.*.csv.gz"))
    game_files = sorted(data_dir.glob("game_data_public.*.csv.gz"))
    replay_files = sorted(data_dir.glob("replay_data_public.*.csv.gz"))

    def extract(path: Path, prefix: str) -> str:
        return path.name[len(prefix):-len(".csv.gz")]

    drafts = {extract(path, "draft_data_public."): path for path in draft_files}
    games = {extract(path, "game_data_public."): path for path in game_files}
    replays = {extract(path, "replay_data_public."): path for path in replay_files}
    common = sorted(set(drafts) & set(games) & set(replays))

    if suffix is not None:
        if suffix not in common:
            raise ValueError(f"No matching draft/game/replay triple for dataset suffix {suffix!r}")
        return [(suffix, drafts[suffix], games[suffix], replays[suffix])]

    if not common:
        raise ValueError(f"No matching draft/game/replay dataset triples found in {data_dir}")

    all_suffixes = set(drafts) | set(games) | set(replays)
    incomplete = sorted(all_suffixes - set(common))
    if incomplete:
        LOG.warning(
            "Ignoring %d dataset(s) without a complete draft/game/replay triple: %s",
            len(incomplete),
            incomplete[:10],
        )

    return [
        (dataset_suffix, drafts[dataset_suffix], games[dataset_suffix], replays[dataset_suffix])
        for dataset_suffix in common
    ]


def resolve_datasets(args: argparse.Namespace) -> list[Paths]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    manual_files = (args.draft_file, args.game_file, args.replay_file)
    if any(manual_files) and not all(manual_files):
        raise ValueError("Pass --draft-file, --game-file, and --replay-file together, or none of them")

    cards_file = Path(args.cards_file).expanduser().resolve() if args.cards_file else data_dir / "cards.csv"
    abilities_file = (
        Path(args.abilities_file).expanduser().resolve()
        if args.abilities_file
        else data_dir / "abilities.csv"
    )
    output = Path(args.output).expanduser().resolve() if args.output else data_dir / "17lands.duckdb"

    if args.game_file:
        draft_file = Path(args.draft_file).expanduser().resolve()
        game_file = Path(args.game_file).expanduser().resolve()
        replay_file = Path(args.replay_file).expanduser().resolve()
        pairs = [(args.dataset_suffix, draft_file, game_file, replay_file)]
    else:
        pairs = discover_dataset_pairs(data_dir, args.dataset_suffix)

    if args.dataset_id and len(pairs) != 1:
        raise ValueError("--dataset-id can only be used when processing one dataset")

    for path in (cards_file, abilities_file):
        if not path.exists():
            raise FileNotFoundError(path)

    resolved = []
    for discovered_suffix, draft_file, game_file, replay_file in pairs:
        for path in (draft_file, game_file, replay_file):
            if not path.exists():
                raise FileNotFoundError(path)
        dataset_id = derive_dataset_id(
            game_file,
            replay_file,
            args.dataset_id or discovered_suffix,
        )
        resolved.append(
            Paths(
                dataset_id=dataset_id,
                draft_file=draft_file,
                game_file=game_file,
                replay_file=replay_file,
                cards_file=cards_file,
                abilities_file=abilities_file,
                output=output,
            )
        )
    return resolved

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-suffix")
    parser.add_argument("--dataset-id")
    parser.add_argument("--draft-file")
    parser.add_argument("--game-file")
    parser.add_argument("--replay-file")
    parser.add_argument("--cards-file")
    parser.add_argument("--abilities-file")
    parser.add_argument("--output")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Base transaction chunk size. Source files are scanned once regardless of this value.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250_000,
        help="Log source-scan progress every N rows for order/draft/game/replay; use 0 to disable.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the in-place progress display and use periodic INFO progress logs instead.",
    )
    parser.add_argument(
        "--s3-backup-every-minutes",
        type=float,
        metavar="MINUTES",
        help=(
            "Opt in to periodic S3 snapshots. After committed chunks, upload at most once "
            "per MINUTES to s3://alistairbegg-personal-projects/mtg_card_embeddings/data/. "
            "A final snapshot is also uploaded after a successful build. Omit this option "
            "to disable all S3 backup activity."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    datasets = resolve_datasets(args)
    LOG.info("Output: %s", datasets[0].output)
    LOG.info("Datasets selected: %d", len(datasets))

    progress = ProgressDisplay(
        len(datasets),
        enabled=(not args.no_progress and sys.stderr.isatty()),
    )
    # Avoid duplicate periodic INFO lines when the live progress renderer is active.
    effective_progress_every = 0 if progress.enabled else args.progress_every

    s3_backup = None
    if args.s3_backup_every_minutes is not None:
        s3_backup = S3BackupManager(
            datasets[0].output,
            args.s3_backup_every_minutes,
        )
        LOG.info(
            "Periodic S3 backup enabled: every %.2f minutes to s3://%s/%s.",
            args.s3_backup_every_minutes,
            S3_BUCKET,
            S3_DATABASE_KEY,
        )

    last_builder: DatabaseBuilder | None = None
    try:
        for index, paths in enumerate(datasets, start=1):
            if not progress.enabled:
                LOG.info("Dataset %d/%d: %s", index, len(datasets), paths.dataset_id)
                LOG.info("Draft CSV: %s", paths.draft_file)
                LOG.info("Game CSV: %s", paths.game_file)
                LOG.info("Replay CSV: %s", paths.replay_file)
            builder = DatabaseBuilder(
                paths=paths,
                batch_size=args.batch_size,
                overwrite=args.overwrite and index == 1,
                progress_every=effective_progress_every,
                s3_backup=s3_backup,
                progress=progress if progress.enabled else None,
                dataset_position=index,
            )
            builder.run()
            last_builder = builder

        # These checks are global over the normalized tables. Running them after every
        # dataset repeatedly rescans a growing multi-set database without adding safety.
        # Run the exact same checks once after all selected datasets have completed.
        if last_builder is not None:
            if progress.enabled:
                progress.begin_dataset(len(datasets), "all datasets")
                progress.start_stage("Checks", start_rows=0, total_rows=1, activity="global integrity checks")
            with duckdb.connect(str(datasets[0].output)) as con:
                last_builder.run_integrity_checks(con, emit_info=not progress.enabled)
            if progress.enabled:
                progress.finish_stage(1)

        if s3_backup is not None:
            # A completed unattended build is only successful once its final
            # database snapshot has been published to S3.
            with duckdb.connect(str(datasets[0].output)) as con:
                uploaded = s3_backup.maybe_upload(
                    con,
                    reason="successful build completion",
                    force=True,
                )

            if not uploaded:
                raise RuntimeError("Final S3 database upload failed")
    finally:
        progress.close()


if __name__ == "__main__":
    main()
