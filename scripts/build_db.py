#!/usr/bin/env python3
"""Build a normalized DuckDB from public 17Lands draft/game/replay data.

For each dataset the builder first imports all game-backed drafts and their pick
history. It then processes the dataset in draft order: for each draft it gets the
games belonging to that draft, and for each game it imports the game row plus its
replay-derived hands, turns, states, zones and events. Source scans are grouped over
a configurable number of drafts so the gzip files are not rescanned once per draft.

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
import gc
import gzip
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
import numpy as np
import pandas as pd

LOG = logging.getLogger("17lands-db")
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
    draft_file VARCHAR,
    game_file VARCHAR NOT NULL,
    replay_file VARCHAR NOT NULL,
    draft_signature VARCHAR,
    game_signature VARCHAR NOT NULL,
    replay_signature VARCHAR NOT NULL,
    next_draft_index BIGINT DEFAULT 0,
    committed_drafts BIGINT DEFAULT 0,
    completed BOOLEAN NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS draft_ingestion_checkpoints (
    dataset_id VARCHAR PRIMARY KEY,
    draft_file VARCHAR NOT NULL,
    draft_signature VARCHAR NOT NULL,
    filter_signature VARCHAR NOT NULL,
    next_source_row BIGINT NOT NULL,
    committed_batches BIGINT NOT NULL,
    completed BOOLEAN NOT NULL,
    updated_at TIMESTAMP NOT NULL
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

DRAFT_OPTIONAL_COLUMNS = {
    "rank",
    "event_match_wins",
    "event_match_losses",
    "user_n_games_bucket",
    "user_game_win_rate_bucket",
    "pick_maindeck_rate",
    "pick_sideboard_in_rate",
}


@dataclass(frozen=True)
class DraftCheckpoint:
    next_source_row: int
    committed_batches: int
    completed: bool


def make_pick_id(draft_id: str, pack_number: int, pick_number: int) -> int:
    return deterministic_bigint("pick", draft_id, pack_number, pick_number)


def set_signature(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_draft_card_columns(columns: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    pack: dict[str, str] = {}
    pool: dict[str, str] = {}
    for column in columns:
        if column.startswith(PACK_PREFIX):
            pack[column] = normalize_card_name(column[len(PACK_PREFIX):])
        elif column.startswith(POOL_PREFIX):
            pool[column] = normalize_card_name(column[len(POOL_PREFIX):])
    return pack, pool


def ensure_draft_schema(con: duckdb.DuckDBPyConnection) -> None:
    # Draft tables are part of SCHEMA_SQL in the standalone builder.
    # Keeping this function makes the draft importer self-contained and explicit.
    required = {
        "drafts",
        "draft_picks",
        "draft_pick_cards",
        "draft_ingestion_checkpoints",
    }
    existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    missing = required - existing
    if missing:
        raise ValueError(f"Database is missing draft tables: {sorted(missing)}")


def sync_draft_card_names(
    con: duckdb.DuckDBPyConnection,
    card_names: Iterable[str],
) -> dict[str, int]:
    names = sorted({normalize_card_name(name) for name in card_names if normalize_card_name(name)})
    mapping = {name: make_card_id(name) for name in names}

    existing = {
        str(name): int(card_id)
        for card_id, name in con.execute("SELECT card_id, card_name FROM cards").fetchall()
    }
    existing_ids = {card_id: name for name, card_id in existing.items()}

    new_rows: list[dict[str, Any]] = []
    for name, card_id in mapping.items():
        current = existing.get(name)
        if current is not None:
            if current != card_id:
                raise ValueError(f"Canonical card ID mismatch for {name!r}")
            continue
        collision = existing_ids.get(card_id)
        if collision is not None and collision != name:
            raise ValueError(f"Deterministic card ID collision: {collision!r} / {name!r}")
        new_rows.append({"card_id": card_id, "card_name": name})

    if new_rows:
        insert_rows(con, "cards", new_rows, int64=("card_id",))
        LOG.info("Draft reference sync: +%d card names", len(new_rows))

    return mapping


def iter_draft_source_batches(
    path: Path,
    usecols: list[str],
    chunk_size: int,
    start_row: int,
) -> Iterator[tuple[int, pd.DataFrame, int]]:
    """Read the very wide draft CSV in bounded-memory chunks.

    Only the few textual columns use pandas StringDtype. The ~1,000 sparse
    pack/pool columns are parsed as float32 and immediately collapsed to int16
    matrices by build_draft_rows(). This avoids millions of Python string objects.
    """
    current = 0
    text_columns = {
        "draft_id", "expansion", "event_type", "draft_time", "rank", "pick"
    }
    dtype: dict[str, Any] = {
        column: "string" for column in text_columns if column in usecols
    }
    for column in usecols:
        if column.startswith(PACK_PREFIX) or column.startswith(POOL_PREFIX):
            dtype[column] = np.float32
    for column in (
        "pack_number",
        "pick_number",
        "event_match_wins",
        "event_match_losses",
        "user_n_games_bucket",
        "user_game_win_rate_bucket",
        "pick_maindeck_rate",
        "pick_sideboard_in_rate",
    ):
        if column in usecols:
            dtype[column] = np.float32

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk_start = current
        chunk_end = current + len(chunk)
        current = chunk_end
        if chunk_end <= start_row:
            continue
        if start_row > chunk_start:
            chunk = chunk.iloc[start_row - chunk_start:]
            chunk_start = start_row
        if len(chunk):
            yield chunk_start, chunk.reset_index(drop=True), chunk_end


def load_or_reset_draft_checkpoint(
    con: duckdb.DuckDBPyConnection,
    *,
    dataset_id: str,
    draft_file: Path,
    draft_signature: str,
    filter_signature: str,
) -> DraftCheckpoint:
    row = con.execute(
        """
        SELECT draft_signature, filter_signature, next_source_row,
               committed_batches, completed
        FROM draft_ingestion_checkpoints
        WHERE dataset_id = ?
        """,
        [dataset_id],
    ).fetchone()

    if row is None:
        con.execute(
            """
            INSERT INTO draft_ingestion_checkpoints (
                dataset_id, draft_file, draft_signature, filter_signature,
                next_source_row, committed_batches, completed, updated_at
            ) VALUES (?, ?, ?, ?, 0, 0, FALSE, current_timestamp)
            """,
            [dataset_id, str(draft_file), draft_signature, filter_signature],
        )
        con.execute("CHECKPOINT")
        return DraftCheckpoint(0, 0, False)

    saved_file_sig, saved_filter_sig, next_row, batches, completed = row
    if saved_file_sig == draft_signature and saved_filter_sig == filter_signature:
        return DraftCheckpoint(int(next_row), int(batches), bool(completed))

    raise ValueError(
        "Draft checkpoint source/filter no longer matches the current inputs. "
        "Rebuild with --overwrite."
    )


def merge_draft_metadata(
    drafts: dict[str, dict[str, Any]],
    incoming: dict[str, Any],
) -> None:
    draft_id = incoming["draft_id"]
    current = drafts.get(draft_id)
    if current is None:
        drafts[draft_id] = incoming
        return
    for field, value in incoming.items():
        if field == "draft_id" or value is None:
            continue
        previous = current.get(field)
        if previous is None:
            current[field] = value
        elif field == "draft_time":
            if pd.Timestamp(previous) != pd.Timestamp(value):
                raise ValueError(f"Conflicting {field} within draft {draft_id}")
        elif previous != value:
            raise ValueError(
                f"Conflicting {field} within draft {draft_id}: {previous!r} != {value!r}"
            )


def build_draft_rows(
    frame: pd.DataFrame,
    *,
    card_name_to_id: dict[str, int],
    pack_columns: dict[str, str],
    pool_columns: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    drafts: dict[str, dict[str, Any]] = {}
    picks: list[dict[str, Any]] = []
    pick_cards: list[dict[str, Any]] = []

    pack_column_names = list(pack_columns)
    pool_column_names = list(pool_columns)
    pack_names = list(pack_columns.values())
    pool_names = list(pool_columns.values())

    pack_matrix = frame[pack_column_names].fillna(0).to_numpy(dtype=np.int16, copy=True)
    pool_matrix = frame[pool_column_names].fillna(0).to_numpy(dtype=np.int16, copy=True)
    if np.any(pack_matrix < 0) or np.any(pool_matrix < 0):
        raise ValueError("Negative draft pack/pool card count found")

    scalar_columns = [
        column
        for column in (
            "draft_id", "expansion", "event_type", "draft_time", "rank",
            "event_match_wins", "event_match_losses", "user_n_games_bucket",
            "user_game_win_rate_bucket", "pack_number", "pick_number", "pick",
            "pick_maindeck_rate", "pick_sideboard_in_rate",
        )
        if column in frame.columns
    ]
    records = frame[scalar_columns].to_dict("records")

    for position, record in enumerate(records):
        draft_id = str(record["draft_id"])
        pack_number = integer_or_none(record.get("pack_number"))
        pick_number = integer_or_none(record.get("pick_number"))
        if pack_number is None or pick_number is None:
            raise ValueError(f"Missing pack/pick number for draft {draft_id}")

        merge_draft_metadata(
            drafts,
            {
                "draft_id": draft_id,
                "expansion": text_or_none(record.get("expansion")),
                "event_type": text_or_none(record.get("event_type")),
                "draft_time": timestamp_or_none(record.get("draft_time")),
                "rank": text_or_none(record.get("rank")),
                "event_match_wins": integer_or_none(record.get("event_match_wins")),
                "event_match_losses": integer_or_none(record.get("event_match_losses")),
                "user_n_games_bucket": integer_or_none(record.get("user_n_games_bucket")),
                "user_game_win_rate_bucket": numeric_or_none(record.get("user_game_win_rate_bucket")),
            },
        )

        picked_name = text_or_none(record.get("pick"))
        if picked_name is None:
            raise ValueError(
                f"Missing selected card for {draft_id} pack={pack_number} pick={pick_number}"
            )
        picked_name = normalize_card_name(picked_name)
        picked_id = card_name_to_id.get(picked_name)
        if picked_id is None:
            raise ValueError(f"Draft pick card {picked_name!r} has no canonical card ID")
        pick_id = make_pick_id(draft_id, pack_number, pick_number)

        pack_row = pack_matrix[position]
        pool_row = pool_matrix[position]
        card_counts: dict[int, list[int]] = {}
        picked_offered = False

        for index in np.flatnonzero(pack_row):
            i = int(index)
            name = pack_names[i]
            count = int(pack_row[i])
            card_id = card_name_to_id[name]
            card_counts.setdefault(card_id, [0, 0])[0] = count
            if name == picked_name:
                picked_offered = True

        if not picked_offered:
            raise ValueError(
                f"Selected card {picked_name!r} not offered for {draft_id} "
                f"pack={pack_number} pick={pick_number}"
            )

        for index in np.flatnonzero(pool_row):
            i = int(index)
            name = pool_names[i]
            count = int(pool_row[i])
            card_id = card_name_to_id[name]
            card_counts.setdefault(card_id, [0, 0])[1] = count

        picks.append(
            {
                "pick_id": pick_id,
                "draft_id": draft_id,
                "pack_number": pack_number,
                "pick_number": pick_number,
                "card_id": picked_id,
                "pick_maindeck_rate": numeric_or_none(record.get("pick_maindeck_rate")),
                "pick_sideboard_in_rate": numeric_or_none(record.get("pick_sideboard_in_rate")),
            }
        )
        pick_cards.extend(
            {
                "pick_id": pick_id,
                "card_id": card_id,
                "pack_count": counts[0],
                "pool_count": counts[1],
            }
            for card_id, counts in card_counts.items()
        )

    return drafts, picks, pick_cards


def upsert_draft_metadata(
    con: duckdb.DuckDBPyConnection,
    drafts: dict[str, dict[str, Any]],
) -> None:
    fields = (
        "expansion",
        "event_type",
        "draft_time",
        "rank",
        "event_match_wins",
        "event_match_losses",
        "user_n_games_bucket",
        "user_game_win_rate_bucket",
    )

    for draft_id, row in drafts.items():
        existing = con.execute(
            f"SELECT {', '.join(fields)} FROM drafts WHERE draft_id = ?",
            [draft_id],
        ).fetchone()

        if existing is None:
            con.execute(
                f"INSERT INTO drafts (draft_id, {', '.join(fields)}) "
                f"VALUES (?, {', '.join('?' for _ in fields)})",
                [draft_id, *(row[field] for field in fields)],
            )
            continue

        updates: dict[str, Any] = {}
        for stored, field in zip(existing, fields, strict=True):
            source = row[field]
            if source is None:
                continue
            if stored is None:
                updates[field] = source
                continue
            equal = pd.Timestamp(stored) == pd.Timestamp(source) if field == "draft_time" else stored == source
            if not equal:
                raise ValueError(
                    f"Draft metadata changed for {draft_id}: {field} {stored!r} != {source!r}"
                )

        if updates:
            assignments = ", ".join(f"{field} = ?" for field in updates)
            con.execute(
                f"UPDATE drafts SET {assignments} WHERE draft_id = ?",
                [*updates.values(), draft_id],
            )


def replace_draft_picks(
    con: duckdb.DuckDBPyConnection,
    picks: list[dict[str, Any]],
    pick_cards: list[dict[str, Any]],
) -> None:
    if not picks:
        return

    pick_frame = rows_to_frame(
        picks,
        int64=("pick_id", "card_id"),
        int16=("pack_number", "pick_number"),
    )
    con.register("_draft_pick_upserts", pick_frame)
    try:
        con.execute(
            "DELETE FROM draft_pick_cards WHERE pick_id IN "
            "(SELECT pick_id FROM _draft_pick_upserts)"
        )
        con.execute(
            "DELETE FROM draft_picks WHERE pick_id IN "
            "(SELECT pick_id FROM _draft_pick_upserts)"
        )
        columns = list(pick_frame.columns)
        quoted = ", ".join(f'"{column}"' for column in columns)
        con.execute(
            f"INSERT INTO draft_picks ({quoted}) SELECT {quoted} FROM _draft_pick_upserts"
        )
    finally:
        con.unregister("_draft_pick_upserts")

    insert_rows(
        con,
        "draft_pick_cards",
        pick_cards,
        int64=("pick_id", "card_id"),
        int16=("pack_count", "pool_count"),
    )


def ingest_draft_file(
    con: duckdb.DuckDBPyConnection,
    draft_file: Path,
    *,
    dataset_id: str,
    allowed_draft_ids: set[str],
    chunk_size: int = 500,
    max_batches: int | None = None,
    require_all: bool = True,
) -> bool:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    ensure_draft_schema(con)
    header = read_header(draft_file)
    missing = DRAFT_REQUIRED_COLUMNS - set(header)
    if missing:
        raise ValueError(f"{draft_file.name} is missing required columns: {sorted(missing)}")

    pack_columns, pool_columns = parse_draft_card_columns(header)
    if not pack_columns or not pool_columns:
        raise ValueError("Draft CSV must contain both pack_card_* and pool_* columns")

    card_name_to_id = sync_draft_card_names(
        con,
        set(pack_columns.values()) | set(pool_columns.values()),
    )

    usecols = [
        column
        for column in header
        if column in DRAFT_REQUIRED_COLUMNS
        or column in DRAFT_OPTIONAL_COLUMNS
        or column in pack_columns
        or column in pool_columns
    ]

    checkpoint = load_or_reset_draft_checkpoint(
        con,
        dataset_id=dataset_id,
        draft_file=draft_file,
        draft_signature=file_signature(draft_file, header),
        filter_signature=set_signature(allowed_draft_ids),
    )
    if checkpoint.completed:
        LOG.info("Draft dataset %s is already complete.", dataset_id)
        return True

    processed = 0
    exhausted = True
    for source_start, source_chunk, source_end in iter_draft_source_batches(
        draft_file,
        usecols,
        chunk_size,
        checkpoint.next_source_row,
    ):
        if max_batches is not None and processed >= max_batches:
            exhausted = False
            break

        mask = source_chunk["draft_id"].astype(str).isin(allowed_draft_ids)
        filtered = source_chunk.loc[mask]

        con.execute("BEGIN")
        try:
            matched = len(filtered)
            if matched:
                pick_names = {
                    normalize_card_name(value)
                    for value in filtered["pick"].dropna().astype(str)
                    if normalize_card_name(value)
                }
                missing_names = pick_names - set(card_name_to_id)
                if missing_names:
                    card_name_to_id.update(sync_draft_card_names(con, missing_names))

                drafts, picks, pick_cards = build_draft_rows(
                    filtered,
                    card_name_to_id=card_name_to_id,
                    pack_columns=pack_columns,
                    pool_columns=pool_columns,
                )
                upsert_draft_metadata(con, drafts)
                replace_draft_picks(con, picks, pick_cards)

            con.execute(
                """
                UPDATE draft_ingestion_checkpoints
                SET next_source_row = ?, committed_batches = ?, draft_file = ?,
                    updated_at = current_timestamp
                WHERE dataset_id = ?
                """,
                [
                    source_end,
                    checkpoint.committed_batches + 1,
                    str(draft_file),
                    dataset_id,
                ],
            )
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise

        processed += 1
        checkpoint = DraftCheckpoint(
            next_source_row=source_end,
            committed_batches=checkpoint.committed_batches + 1,
            completed=False,
        )
        if checkpoint.committed_batches % 20 == 0:
            con.execute("CHECKPOINT")
        LOG.info(
            "Draft batch %d: source rows %d..%d; matched %d rows.",
            checkpoint.committed_batches,
            source_start,
            source_end - 1,
            matched,
        )

        del source_chunk, filtered
        if processed % 20 == 0:
            gc.collect()

    if not exhausted:
        con.execute("CHECKPOINT")
        LOG.info("Stopped draft ingestion cleanly after %d new batches.", processed)
        return False

    if require_all:
        present = {
            str(row[0])
            for row in con.execute("SELECT DISTINCT draft_id FROM draft_picks").fetchall()
        }
        missing_ids = sorted(allowed_draft_ids - present)
        if missing_ids:
            raise ValueError(
                f"Draft source completed but {len(missing_ids)} required draft IDs had no picks; "
                f"first: {missing_ids[:10]}"
            )

    invalid = int(
        con.execute(
            """
            SELECT count(*)
            FROM draft_picks p
            LEFT JOIN draft_pick_cards c
              ON c.pick_id = p.pick_id
             AND c.card_id = p.card_id
             AND c.pack_count > 0
            WHERE c.card_id IS NULL
            """
        ).fetchone()[0]
    )
    if invalid:
        raise ValueError(f"{invalid} selected draft cards are not present in their offered packs")

    con.execute(
        """
        UPDATE draft_ingestion_checkpoints
        SET completed = TRUE, updated_at = current_timestamp
        WHERE dataset_id = ?
        """,
        [dataset_id],
    )
    con.execute("CHECKPOINT")
    LOG.info("Draft dataset %s is complete.", dataset_id)
    return True



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


class DatabaseBuilder:
    def __init__(
        self,
        paths: Paths,
        draft_batch_size: int,
        overwrite: bool,
        max_drafts: int | None,
        replay_progress_every: int,
        draft_chunk_size: int,
        max_draft_batches: int | None,
        memory_limit: str,
        threads: int,
    ) -> None:
        self.paths = paths
        self.draft_batch_size = draft_batch_size
        self.overwrite = overwrite
        self.max_drafts = max_drafts
        self.replay_progress_every = replay_progress_every
        self.draft_chunk_size = draft_chunk_size
        self.max_draft_batches = max_draft_batches
        self.memory_limit = memory_limit
        self.threads = threads

        self.game_columns = read_header(paths.game_file)
        self.replay_columns = read_header(paths.replay_file)
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
        if self.draft_batch_size <= 0:
            raise ValueError("--draft-batch-size must be positive")
        if self.max_drafts is not None and self.max_drafts <= 0:
            raise ValueError("--max-drafts must be positive")
        if self.draft_chunk_size <= 0:
            raise ValueError("--draft-chunk-size must be positive")
        if self.max_draft_batches is not None and self.max_draft_batches <= 0:
            raise ValueError("--max-draft-batches must be positive")
        if self.threads <= 0:
            raise ValueError("--threads must be positive")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:KB|MB|GB|TB)", self.memory_limit, re.I):
            raise ValueError("--memory-limit must look like 768MB, 1GB, etc.")

        validate_required_columns(self.game_columns, self.replay_columns, self.candidate_columns)
        validate_turn_schema(self.turn_columns)
        validate_repeated_turn_schema(self.turn_columns)
        prepare_output(self.paths.output, self.overwrite)

        helper_cards = pd.read_csv(self.paths.cards_file)
        helper_abilities = pd.read_csv(self.paths.abilities_file)
        require_columns(helper_cards.columns, {"id", "name"}, self.paths.cards_file.name)
        require_columns(helper_abilities.columns, {"id", "text"}, self.paths.abilities_file.name)
        self.reference = build_reference_data(helper_cards, helper_abilities, self.game_card_columns)

        draft_signature = file_signature(self.paths.draft_file, read_header(self.paths.draft_file))
        game_signature = file_signature(self.paths.game_file, self.game_columns)
        replay_signature = file_signature(self.paths.replay_file, self.replay_columns)
        ordered_draft_ids = collect_game_draft_ids_ordered(self.paths.game_file)
        allowed_draft_ids = set(ordered_draft_ids)

        con = duckdb.connect(str(self.paths.output))
        temp_dir = self.paths.output.parent / ".duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            memory_limit = self.memory_limit.replace("'", "''")
            temp_sql = str(temp_dir).replace("'", "''")
            con.execute(f"SET memory_limit = '{memory_limit}'")
            con.execute(f"SET threads = {self.threads}")
            con.execute(f"SET temp_directory = '{temp_sql}'")
            con.execute("SET preserve_insertion_order = false")
            LOG.info(
                "DuckDB limits: memory=%s, threads=%d, temp=%s",
                self.memory_limit,
                self.threads,
                temp_dir,
            )
            con.execute(SCHEMA_SQL)
            ensure_draft_schema(con)
            migrate_ingestion_checkpoint_schema(con)
            validate_database_schema(con)
            self.sync_reference_tables(con, helper_cards, helper_abilities)
            del helper_cards, helper_abilities
            gc.collect()

            LOG.info("Game-backed drafts in %s: %d", self.paths.dataset_id, len(ordered_draft_ids))
            draft_complete = ingest_draft_file(
                con,
                self.paths.draft_file,
                dataset_id=self.paths.dataset_id,
                allowed_draft_ids=allowed_draft_ids,
                chunk_size=self.draft_chunk_size,
                max_batches=self.max_draft_batches,
                require_all=True,
            )
            if not draft_complete:
                LOG.info("Draft phase stopped cleanly; rerun the same command before game ingestion.")
                return

            # Draft import can introduce canonical names that are absent from cards.csv.
            self.reference.card_name_to_id.update(
                {str(name): int(card_id) for card_id, name in con.execute(
                    "SELECT card_id, card_name FROM cards"
                ).fetchall()}
            )

            checkpoint = self.load_or_create_checkpoint(
                con,
                draft_signature,
                game_signature,
                replay_signature,
                len(ordered_draft_ids),
            )
            if checkpoint.completed:
                LOG.info("Dataset %s is already complete.", self.paths.dataset_id)
                self.run_integrity_checks(con)
                return

            LOG.info(
                "Resuming %s at draft %d/%d after %d committed drafts.",
                self.paths.dataset_id,
                checkpoint.next_draft_index,
                len(ordered_draft_ids),
                checkpoint.committed_drafts,
            )

            processed = 0
            next_index = checkpoint.next_draft_index
            while next_index < len(ordered_draft_ids):
                if self.max_drafts is not None and processed >= self.max_drafts:
                    break

                remaining_limit = None if self.max_drafts is None else self.max_drafts - processed
                scan_size = self.draft_batch_size
                if remaining_limit is not None:
                    scan_size = min(scan_size, remaining_limit)
                batch_ids = ordered_draft_ids[next_index:next_index + scan_size]
                if not batch_ids:
                    break

                LOG.info(
                    "Loading source rows for drafts %d..%d (%d drafts).",
                    next_index,
                    next_index + len(batch_ids) - 1,
                    len(batch_ids),
                )
                game_batch = extract_game_rows_for_drafts(
                    self.paths.game_file,
                    self.game_usecols(),
                    batch_ids,
                )
                ordered_keys = game_keys(game_batch)
                replay_batch, scanned = extract_replay_rows(
                    self.paths.replay_file,
                    self.replay_columns,
                    self.replay_usecols(),
                    ordered_keys,
                    self.replay_progress_every,
                )
                LOG.info("Replay scan found %d games after %d source rows.", len(replay_batch), scanned)

                # extract_replay_rows returns rows in exactly the same natural-key
                # order as game_batch. Slice both frames by position instead of
                # materialising the wide replay frame as Python dictionaries.
                draft_values = game_batch["draft_id"].astype(str).to_numpy()

                for draft_id in batch_ids:
                    if self.max_drafts is not None and processed >= self.max_drafts:
                        break
                    positions = np.flatnonzero(draft_values == draft_id)
                    if not len(positions):
                        raise ValueError(f"No game rows found for expected draft {draft_id}")
                    draft_games = game_batch.iloc[positions].reset_index(drop=True)
                    draft_replays = replay_batch.iloc[positions].reset_index(drop=True)

                    current = Checkpoint(
                        next_draft_index=next_index,
                        committed_drafts=checkpoint.committed_drafts + processed,
                        completed=False,
                    )
                    self.process_draft(con, draft_id, draft_games, draft_replays, current)
                    next_index += 1
                    processed += 1

                del game_batch, replay_batch, draft_values
                gc.collect()

            final_checkpoint = Checkpoint(
                next_draft_index=next_index,
                committed_drafts=checkpoint.committed_drafts + processed,
                completed=False,
            )
            if next_index >= len(ordered_draft_ids):
                self.mark_complete(con, final_checkpoint)
                LOG.info("Dataset %s is complete.", self.paths.dataset_id)
            else:
                LOG.info("Stopped cleanly after %d new drafts; rerun the same command to resume.", processed)

            self.run_integrity_checks(con)
        finally:
            con.close()

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
        draft_count: int,
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
            # Older databases may still have the legacy NOT NULL game-row
            # checkpoint columns. Populate them with zero when present so a new
            # dataset can use the draft-oriented builder in the same database.
            checkpoint_columns = {
                info[1] for info in con.execute("PRAGMA table_info('ingestion_checkpoints')").fetchall()
            }
            columns = [
                "dataset_id", "draft_file", "game_file", "replay_file",
                "draft_signature", "game_signature", "replay_signature",
                "next_draft_index", "committed_drafts", "completed", "updated_at",
            ]
            values: list[Any] = [
                self.paths.dataset_id,
                str(self.paths.draft_file),
                str(self.paths.game_file),
                str(self.paths.replay_file),
                draft_signature,
                game_signature,
                replay_signature,
                0,
                0,
                False,
                pd.Timestamp.now(),
            ]
            if "next_game_row" in checkpoint_columns:
                columns.append("next_game_row")
                values.append(0)
            if "committed_batches" in checkpoint_columns:
                columns.append("committed_batches")
                values.append(0)
            placeholders = ", ".join("?" for _ in columns)
            con.execute(
                f"INSERT INTO ingestion_checkpoints ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
            con.execute("CHECKPOINT")
            return Checkpoint(0, 0, False)

        saved_draft, saved_game, saved_replay, next_index, committed, completed = row
        if saved_draft is None and completed:
            # Completed databases built by the older game-row checkpoint builder can
            # be adopted safely: their games/events are already present and the new
            # draft phase has just been run above.
            con.execute(
                """
                UPDATE ingestion_checkpoints
                SET draft_file = ?, draft_signature = ?, next_draft_index = ?,
                    committed_drafts = ?, updated_at = current_timestamp
                WHERE dataset_id = ?
                """,
                [str(self.paths.draft_file), draft_signature, draft_count, draft_count, self.paths.dataset_id],
            )
            con.execute("CHECKPOINT")
            return Checkpoint(draft_count, draft_count, True)

        if saved_draft is None and not completed:
            raise ValueError(
                "This dataset has an unfinished checkpoint from the older game-row builder. "
                "Finish it with the old builder or rebuild this dataset with --overwrite."
            )

        if (saved_draft, saved_game, saved_replay) != (draft_signature, game_signature, replay_signature):
            raise ValueError(
                "Checkpoint file signatures do not match the current draft/game/replay files. "
                "Use the original files or rebuild with --overwrite."
            )
        return Checkpoint(int(next_index or 0), int(committed or 0), bool(completed))

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

        # A database CHECKPOINT for every draft is unnecessarily expensive. WAL
        # commits already make each draft resumable; bound the WAL periodically.
        committed = checkpoint.committed_drafts + 1
        if committed % 100 == 0:
            con.execute("CHECKPOINT")
        LOG.info(
            "Draft %s committed: %d games; checkpoint draft index %d.",
            draft_id,
            len(game_batch),
            checkpoint.next_draft_index + 1,
        )

    def build_draft_rows(self, game_batch: pd.DataFrame) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for record in game_batch.to_dict("records"):
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
        build_columns = [
            item["column"]
            for item in self.game_card_columns
            if item["stat"] in {"deck", "sideboard"}
        ]
        if not build_columns:
            return [None] * len(game_batch), {}

        matrix = numeric_count_matrix(game_batch, build_columns)
        column_info = {
            item["column"]: item
            for item in self.game_card_columns
            if item["stat"] in {"deck", "sideboard"}
        }
        compositions: dict[tuple[str, int], dict[int, tuple[int, int]]] = {}
        build_ids: list[int | None] = []

        for position, record in enumerate(game_batch[["draft_id", "build_index"]].to_dict("records")):
            draft_id = str(record["draft_id"])
            build_index = integer_or_none(record.get("build_index"))
            if build_index is None:
                build_ids.append(None)
                continue

            card_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
            for column, raw_count in matrix.iloc[position].items():
                count = int(raw_count)
                if count == 0:
                    continue
                info = column_info[column]
                card_id = self.reference.card_name_to_id[info["card_name"]]
                slot = 0 if info["stat"] == "deck" else 1
                card_counts[card_id][slot] = count
            composition = {
                card_id: (counts[0], counts[1])
                for card_id, counts in card_counts.items()
            }
            key = (draft_id, build_index)
            previous = compositions.get(key)
            if previous is not None and previous != composition:
                raise ValueError(f"Deck composition changed within batch for {key}")
            compositions[key] = composition
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
                raise ValueError(f"Deterministic build ID mismatch for {(draft_id, build_index)}")
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
                raise ValueError(f"Deck composition changed for {(draft_id, build_index)}")

    def build_game_rows(
        self,
        game_batch: pd.DataFrame,
        game_ids: list[int],
        build_ids: list[int | None],
    ) -> list[dict[str, Any]]:
        rows = []
        for position, record in enumerate(game_batch.to_dict("records")):
            rows.append(
                {
                    "game_id": game_ids[position],
                    "draft_id": str(record["draft_id"]),
                    "build_id": build_ids[position],
                    "match_number": integer_or_none(record.get("match_number")),
                    "game_number": integer_or_none(record.get("game_number")),
                    "game_time": timestamp_or_none(record.get("game_time")),
                    "user_on_play": parse_bool(record.get("on_play")),
                    "user_won": parse_bool(record.get("won")),
                    "source_num_turns": integer_or_none(record.get("num_turns")),
                }
            )
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
        stats: dict[tuple[int, int], dict[str, int]] = defaultdict(
            lambda: {"opening_hand": 0, "drawn": 0, "tutored": 0}
        )
        for item in self.game_card_columns:
            stat = item["stat"]
            if stat not in GAME_STAT_PREFIXES:
                continue
            values = pd.to_numeric(game_batch[item["column"]], errors="coerce").fillna(0)
            positions = np.flatnonzero(values.to_numpy())
            if not len(positions):
                continue
            card_id = self.reference.card_name_to_id[item["card_name"]]
            for position in positions:
                count = int(values.iloc[position])
                if count < 0:
                    raise ValueError(f"Negative card count in {item['column']}")
                stats[(game_ids[position], card_id)][stat] = count

        return [
            {
                "game_id": game_id,
                "card_id": card_id,
                "opening_hand_count": values["opening_hand"],
                "drawn_count": values["drawn"],
                "tutored_count": values["tutored"],
            }
            for (game_id, card_id), values in stats.items()
            if any(values.values())
        ]

    def find_turns(
        self,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> set[tuple[int, str, int]]:
        active: set[tuple[int, str, int]] = set()
        for column, (source_side, source_turn_index, _) in self.turn_columns.items():
            if column not in replay_batch.columns:
                continue
            mask = meaningful_mask(replay_batch[column])
            for position in np.flatnonzero(mask.to_numpy()):
                active.add((game_ids[position], source_side, source_turn_index))
        return active

    def build_turn_rows(
        self,
        active: set[tuple[int, str, int]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "turn_id": make_turn_id(game_id, source_side, source_turn_index),
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
    ) -> list[dict[str, Any]]:
        assert self.reference is not None
        rows: list[dict[str, Any]] = []
        for column, rule in self.event_columns.items():
            if column not in replay_batch.columns:
                continue
            source_side, source_turn_index, _ = self.turn_columns[column]
            mask = meaningful_mask(replay_batch[column])
            for position in np.flatnonzero(mask.to_numpy()):
                game_id = game_ids[position]
                value = replay_batch.iloc[position][column]
                source_turn_id = make_turn_id(game_id, source_side, source_turn_index)
                # Generic cast/draw/discard buckets identify who acted, but not always whose turn it was.
                actual_turn_id = source_turn_id if rule.exact_turn else None
                actor = resolve_player(rule.actor, source_side)
                affected = resolve_player(rule.affected, source_side)

                if rule.payload == "card":
                    for ordinal, token in enumerate(split_ids(value), start=1):
                        arena_id = parse_source_id(token)
                        rows.append(
                            event_row(
                                game_id=game_id,
                                source_turn_id=source_turn_id,
                                actual_turn_id=actual_turn_id,
                                source_field=column,
                                source_ordinal=ordinal,
                                event_type=rule.event_type,
                                actor_is_user=actor,
                                affected_is_user=affected,
                                source_arena_card_id=arena_id,
                                card_id=self.reference.arena_to_card_id.get(arena_id),
                            )
                        )
                elif rule.payload == "ability":
                    for ordinal, token in enumerate(split_ids(value), start=1):
                        ability_id = parse_source_id(token)
                        rows.append(
                            event_row(
                                game_id=game_id,
                                source_turn_id=source_turn_id,
                                actual_turn_id=actual_turn_id,
                                source_field=column,
                                source_ordinal=ordinal,
                                event_type=rule.event_type,
                                actor_is_user=actor,
                                affected_is_user=affected,
                                source_ability_id=ability_id,
                                ability_id=ability_id if ability_id in self.reference.ability_ids else None,
                            )
                        )
                elif rule.payload == "numeric":
                    numeric = numeric_or_none(value)
                    if numeric is None or numeric == 0:
                        continue
                    rows.append(
                        event_row(
                            game_id=game_id,
                            source_turn_id=source_turn_id,
                            actual_turn_id=actual_turn_id,
                            source_field=column,
                            source_ordinal=1,
                            event_type=rule.event_type,
                            actor_is_user=actor,
                            affected_is_user=affected,
                            numeric_value=numeric,
                        )
                    )
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

        for position, record in enumerate(replay_batch.to_dict("records")):
            game_id = game_ids[position]
            mulligans = integer_or_none(record.get("num_mulligans"))
            candidates = [
                (attempt, cards)
                for attempt, column in enumerate(self.candidate_columns, start=1)
                if (cards := split_ids(record.get(column)))
            ]
            if mulligans is None or len(candidates) != mulligans + 1:
                raise ValueError(f"Candidate-hand count mismatch for game {game_id}")
            if any(len(cards) != 7 for _, cards in candidates):
                raise ValueError(f"Non-seven-card candidate hand for game {game_id}")

            # The final candidate is the kept seven and still includes cards later put on the bottom.
            final_attempt, final_cards = candidates[-1]
            if Counter(final_cards) != Counter(split_ids(record.get("opening_hand"))):
                raise ValueError(f"opening_hand differs from final candidate for game {game_id}")

            for attempt, cards in candidates:
                hand_id = make_hand_id(game_id, attempt)
                hand_rows.append(
                    {
                        "hand_id": hand_id,
                        "game_id": game_id,
                        "attempt_number": attempt,
                        "is_final_candidate": attempt == final_attempt,
                    }
                )
                for slot_number, token in enumerate(cards, start=1):
                    arena_id = parse_source_id(token)
                    card_rows.append(
                        {
                            "hand_id": hand_id,
                            "slot_number": slot_number,
                            "source_arena_card_id": arena_id,
                            "card_id": self.reference.arena_to_card_id.get(arena_id),
                        }
                    )
        return hand_rows, card_rows

    def build_turn_state_and_zones(
        self,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
        active_turns: set[tuple[int, str, int]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assert self.reference is not None
        positions = {game_id: position for position, game_id in enumerate(game_ids)}
        state_rows: list[dict[str, Any]] = []
        zone_rows: list[dict[str, Any]] = []

        for game_id, source_side, source_turn_index in sorted(active_turns):
            record = replay_batch.iloc[positions[game_id]]
            turn_id = make_turn_id(game_id, source_side, source_turn_index)
            stem = f"{source_side}_turn_{source_turn_index}_"

            for player_is_user, player in ((True, "user"), (False, "oppo")):
                life = numeric_or_none(record.get(f"{stem}eot_{player}_life"))
                poison = numeric_or_none(record.get(f"{stem}eot_{player}_poison_counters"))
                mana_spent = numeric_or_none(record.get(f"{stem}{player}_mana_spent"))
                if player_is_user:
                    raw_hand = record.get(f"{stem}eot_user_cards_in_hand")
                    hand_size = None if is_missing_or_blank(raw_hand) else len(split_ids(raw_hand))
                else:
                    hand_size = integer_or_none(record.get(f"{stem}eot_oppo_cards_in_hand"))
                if any(value is not None for value in (life, poison, mana_spent, hand_size)):
                    state_rows.append(
                        {
                            "turn_id": turn_id,
                            "player_is_user": player_is_user,
                            "life": life,
                            "poison_counters": poison,
                            "hand_size": hand_size,
                            "mana_spent": mana_spent,
                        }
                    )

            for suffix, (owner_is_user, zone) in ZONE_FIELDS.items():
                counts = Counter(
                    parse_source_id(token)
                    for token in split_ids(record.get(stem + suffix))
                )
                for arena_id, quantity in counts.items():
                    zone_rows.append(
                        {
                            "turn_id": turn_id,
                            "owner_is_user": owner_is_user,
                            "zone": zone,
                            "source_arena_card_id": arena_id,
                            "card_id": self.reference.arena_to_card_id.get(arena_id),
                            "quantity": quantity,
                        }
                    )
        return state_rows, zone_rows

    def build_total_rows(
        self,
        replay_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> list[dict[str, Any]]:
        rows = []
        for position, record in enumerate(replay_batch.to_dict("records")):
            for is_user, side in ((True, "user"), (False, "oppo")):
                row: dict[str, Any] = {"game_id": game_ids[position], "is_user": is_user}
                for output_name, source_suffix in TOTAL_FIELDS.items():
                    source_column = f"{side}_{source_suffix}"
                    row[output_name] = (
                        numeric_or_none(record.get(source_column))
                        if source_column in replay_batch.columns
                        else None
                    )
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

    def run_integrity_checks(self, con: duckdb.DuckDBPyConnection) -> None:
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
        "draft_ingestion_checkpoints": {"dataset_id", "next_source_row", "completed"},
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


def extract_replay_rows(
    replay_file: Path,
    replay_header: list[str],
    wanted_columns: list[str],
    ordered_keys: list[tuple[str, int, int]],
    progress_every: int,
) -> tuple[pd.DataFrame, int]:
    target = set(ordered_keys)
    if len(target) != len(ordered_keys):
        raise ValueError("Duplicate natural game key requested from replay data")

    index = {column: position for position, column in enumerate(replay_header)}
    missing_columns = [column for column in wanted_columns if column not in index]
    if missing_columns:
        raise ValueError(f"Replay CSV missing requested columns: {missing_columns[:20]}")
    key_indices = [index[column] for column in GAME_KEY]
    wanted_indices = [(column, index[column]) for column in wanted_columns]

    # Store compact value lists rather than one dict per wide replay row.
    found: dict[tuple[str, int, int], list[str]] = {}
    rows_scanned = 0
    with open_csv_text(replay_file) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != replay_header:
            raise ValueError("Replay header changed between startup and batch scan")

        for row in reader:
            rows_scanned += 1
            if len(row) != len(replay_header):
                raise ValueError(
                    f"Malformed replay CSV row {rows_scanned}: {len(row)} fields, "
                    f"expected {len(replay_header)}"
                )
            key = (
                row[key_indices[0]],
                parse_key_integer(row[key_indices[1]]),
                parse_key_integer(row[key_indices[2]]),
            )
            if key in target and key not in found:
                found[key] = [row[position] for _, position in wanted_indices]
                if len(found) == len(target):
                    break
            if progress_every > 0 and rows_scanned % progress_every == 0:
                LOG.info(
                    "Replay scan progress: %d rows; found %d/%d target games",
                    rows_scanned,
                    len(found),
                    len(target),
                )

    missing = [key for key in ordered_keys if key not in found]
    if missing:
        raise ValueError(
            f"Replay scan ended with {len(missing)} game rows missing; first keys: {missing[:10]}"
        )
    frame = pd.DataFrame([found[key] for key in ordered_keys], columns=wanted_columns)
    return frame, rows_scanned


def collect_game_draft_ids_ordered(game_file: Path, chunk_size: int = 100_000) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for chunk in pd.read_csv(
        game_file,
        usecols=["draft_id"],
        dtype="string",
        chunksize=chunk_size,
        low_memory=False,
    ):
        for value in chunk["draft_id"].dropna().astype(str):
            if value not in seen:
                seen.add(value)
                result.append(value)
    if not result:
        raise ValueError(f"No draft IDs found in game file {game_file}")
    return result


def extract_game_rows_for_drafts(
    game_file: Path,
    usecols: list[str],
    ordered_draft_ids: list[str],
    chunk_size: int = 500,
) -> pd.DataFrame:
    target = set(ordered_draft_ids)
    parts: list[pd.DataFrame] = []

    text_columns = {
        "draft_id", "expansion", "event_type", "draft_time", "game_time",
        "rank", "opp_rank", "main_colors", "splash_colors", "opp_colors",
    }
    dtype: dict[str, Any] = {
        column: "string" for column in text_columns if column in usecols
    }
    for column in usecols:
        if any(column.startswith(prefix + "_") for prefix in GAME_CARD_PREFIXES):
            dtype[column] = np.float32

    for chunk in pd.read_csv(
        game_file,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False,
    ):
        selected = chunk.loc[chunk["draft_id"].astype(str).isin(target)]
        if not selected.empty:
            parts.append(selected.copy())

    if not parts:
        raise ValueError(f"No game rows found for requested drafts: {ordered_draft_ids[:10]}")

    frame = pd.concat(parts, ignore_index=True)
    found = set(frame["draft_id"].astype(str))
    missing = [draft_id for draft_id in ordered_draft_ids if draft_id not in found]
    if missing:
        raise ValueError(f"Game CSV is missing requested drafts: {missing[:10]}")

    order = {draft_id: index for index, draft_id in enumerate(ordered_draft_ids)}
    frame["_draft_order"] = frame["draft_id"].astype(str).map(order)
    frame["_source_order"] = np.arange(len(frame))
    frame = frame.sort_values(["_draft_order", "_source_order"])
    return frame.drop(columns=["_draft_order", "_source_order"]).reset_index(drop=True)


def migrate_ingestion_checkpoint_schema(con: duckdb.DuckDBPyConnection) -> None:
    existing = {row[1] for row in con.execute("PRAGMA table_info('ingestion_checkpoints')").fetchall()}
    additions = {
        "draft_file": "VARCHAR",
        "draft_signature": "VARCHAR",
        "next_draft_index": "BIGINT DEFAULT 0",
        "committed_drafts": "BIGINT DEFAULT 0",
    }
    for column, definition in additions.items():
        if column not in existing:
            con.execute(f"ALTER TABLE ingestion_checkpoints ADD COLUMN {column} {definition}")

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
) -> ReferenceData:
    helper_names = {
        normalize_card_name(name)
        for name in helper_cards["name"].dropna().astype(str)
        if normalize_card_name(name)
    }
    game_names = {item["card_name"] for item in game_card_columns}
    names = sorted(helper_names | game_names)
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
    if is_missing_or_blank(value):
        return []
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return []
    return [token.strip() for token in text.split("|") if token.strip()]


def parse_source_id(token: Any) -> int:
    text = str(token).strip()
    try:
        return int(text)
    except ValueError:
        return int(float(text))


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
        if isinstance(a, pd.Timestamp) or isinstance(b, pd.Timestamp):
            if pd.Timestamp(a) != pd.Timestamp(b):
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


def open_csv_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


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
        "--draft-batch-size", "--batch-size",
        dest="draft_batch_size",
        type=int,
        default=100,
        help="Number of drafts whose game/replay source rows are gathered in one source scan.",
    )
    parser.add_argument(
        "--draft-chunk-size",
        type=int,
        default=500,
        help="Rows per source chunk while importing the very wide draft-pick CSV.",
    )
    parser.add_argument(
        "--max-draft-batches",
        type=int,
        help="Process at most this many draft-source chunks, then stop before games.",
    )
    parser.add_argument(
        "--max-drafts", "--max-batches",
        dest="max_drafts",
        type=int,
        help="Process at most this many game-backed drafts, then stop cleanly.",
    )
    parser.add_argument(
        "--replay-progress-every",
        type=int,
        default=500_000,
        help="Log replay-scan progress every N rows; use 0 to disable.",
    )
    parser.add_argument(
        "--memory-limit",
        default="512MB",
        help="DuckDB memory limit. 512MB leaves headroom for Python on a 2GB EC2 instance.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB worker threads; fewer threads reduce peak memory.",
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

    for index, paths in enumerate(datasets, start=1):
        LOG.info("Dataset %d/%d: %s", index, len(datasets), paths.dataset_id)
        LOG.info("Draft CSV: %s", paths.draft_file)
        LOG.info("Game CSV: %s", paths.game_file)
        LOG.info("Replay CSV: %s", paths.replay_file)
        DatabaseBuilder(
            paths=paths,
            draft_batch_size=args.draft_batch_size,
            overwrite=args.overwrite and index == 1,
            max_drafts=args.max_drafts,
            replay_progress_every=args.replay_progress_every,
            draft_chunk_size=args.draft_chunk_size,
            max_draft_batches=args.max_draft_batches,
            memory_limit=args.memory_limit,
            threads=args.threads,
        ).run()


if __name__ == "__main__":
    main()
