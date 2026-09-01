#!/usr/bin/env python3
"""Build a normalized DuckDB from public 17Lands draft/game/replay data.

For each dataset the builder scans each large source only once for normalization:
it establishes first-seen draft order from game_data, imports remaining draft picks in
one draft_data pass, imports remaining games/builds in one game_data pass, then imports
replay-derived hands, turns, states, zones and events in one replay_data pass.

The existing next_draft_index checkpoint remains authoritative, so databases created by
the previous draft-batched builder can resume without rebuilding completed drafts.
Additional source-phase checkpoints make the single-pass scans resumable after failure.

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
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    text = str(value).strip()
    if text == "" or text.lower() in {"0", "0.0", "nan", "none", "<na>"}:
        return 0
    return int(float(text))


def upsert_draft_metadata(
    con: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
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
    for row in rows:
        draft_id = row["draft_id"]
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
        for stored, field in zip(existing, fields, strict=True):
            source = row[field]
            if source is None:
                continue
            equal = values_equal((stored,), (source,))
            if stored is not None and not equal:
                raise ValueError(
                    f"Draft metadata changed for {draft_id}: "
                    f"{field} {stored!r} != {source!r}"
                )
            if stored is None:
                con.execute(
                    f"UPDATE drafts SET {field} = ? WHERE draft_id = ?",
                    [source, draft_id],
                )


def clear_partial_draft_picks(
    con: duckdb.DuckDBPyConnection,
    draft_ids: set[str],
) -> None:
    frame = pd.DataFrame({"draft_id": sorted(draft_ids)})
    con.register("_target_drafts", frame)
    try:
        con.execute(
            "DELETE FROM draft_pick_cards WHERE pick_id IN ("
            "SELECT p.pick_id FROM draft_picks p "
            "JOIN _target_drafts d USING (draft_id))"
        )
        con.execute(
            "DELETE FROM draft_picks WHERE draft_id IN "
            "(SELECT draft_id FROM _target_drafts)"
        )
    finally:
        con.unregister("_target_drafts")


def clear_partial_replay_data(
    con: duckdb.DuckDBPyConnection,
    dataset_id: str,
) -> None:
    """Remove replay-derived rows only for drafts not yet fully checkpointed."""
    game_filter = """
        SELECT g.game_id
        FROM games g
        JOIN ingestion_draft_status s ON s.draft_id = g.draft_id
        WHERE s.dataset_id = ? AND s.completed = FALSE
    """
    turn_filter = f"SELECT turn_id FROM turns WHERE game_id IN ({game_filter})"
    hand_filter = f"SELECT hand_id FROM candidate_hands WHERE game_id IN ({game_filter})"

    con.execute("BEGIN")
    try:
        con.execute(f"DELETE FROM events WHERE game_id IN ({game_filter})", [dataset_id])
        con.execute(
            f"DELETE FROM turn_player_state WHERE turn_id IN ({turn_filter})",
            [dataset_id],
        )
        con.execute(
            f"DELETE FROM turn_zone_cards WHERE turn_id IN ({turn_filter})",
            [dataset_id],
        )
        con.execute(f"DELETE FROM turns WHERE game_id IN ({game_filter})", [dataset_id])
        con.execute(
            f"DELETE FROM candidate_hand_cards WHERE hand_id IN ({hand_filter})",
            [dataset_id],
        )
        con.execute(
            f"DELETE FROM candidate_hands WHERE game_id IN ({game_filter})",
            [dataset_id],
        )
        con.execute(
            f"DELETE FROM game_player_totals WHERE game_id IN ({game_filter})",
            [dataset_id],
        )
        con.execute(
            f"""
            UPDATE game_players
            SET n_games_bucket = NULL, game_win_rate_bucket = NULL
            WHERE is_user = TRUE AND game_id IN ({game_filter})
            """,
            [dataset_id],
        )
        con.execute(
            """
            UPDATE ingestion_draft_status
            SET replay_games_processed = 0
            WHERE dataset_id = ? AND completed = FALSE
            """,
            [dataset_id],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


def clear_partial_game_data(
    con: duckdb.DuckDBPyConnection,
    dataset_id: str,
) -> None:
    """Remove normalized game data for only the uncheckpointed draft suffix."""
    clear_partial_replay_data(con, dataset_id)
    game_filter = """
        SELECT g.game_id
        FROM games g
        JOIN ingestion_draft_status s ON s.draft_id = g.draft_id
        WHERE s.dataset_id = ? AND s.completed = FALSE
    """
    build_filter = """
        SELECT b.build_id
        FROM deck_builds b
        JOIN ingestion_draft_status s ON s.draft_id = b.draft_id
        WHERE s.dataset_id = ? AND s.completed = FALSE
    """
    con.execute("BEGIN")
    try:
        con.execute(f"DELETE FROM game_card_stats WHERE game_id IN ({game_filter})", [dataset_id])
        con.execute(f"DELETE FROM game_players WHERE game_id IN ({game_filter})", [dataset_id])
        con.execute(f"DELETE FROM games WHERE game_id IN ({game_filter})", [dataset_id])
        con.execute(
            f"DELETE FROM deck_build_cards WHERE build_id IN ({build_filter})",
            [dataset_id],
        )
        con.execute(
            f"DELETE FROM deck_builds WHERE build_id IN ({build_filter})",
            [dataset_id],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise


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


class DatabaseBuilder:
    def __init__(
        self,
        paths: Paths,
        batch_size: int,
        overwrite: bool,
        replay_progress_every: int,
    ) -> None:
        self.paths = paths
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.replay_progress_every = replay_progress_every

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
                LOG.info("Dataset %s is already complete.", self.paths.dataset_id)
                self.run_integrity_checks(con)
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
            LOG.info("Dataset %s is complete.", self.paths.dataset_id)
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
        """Build first-seen draft order and expected game counts in one game-file pass."""
        phase = self.load_or_create_source_checkpoint(
            con, "order", self.paths.game_file, game_signature
        )
        if phase.completed:
            return

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

        index = {column: position for position, column in enumerate(self.game_columns)}
        draft_position = index["draft_id"]
        pending_new: list[tuple[str, int, str, bool]] = []
        pending_counts: Counter[str] = Counter()
        rows_since_commit = 0
        source_row = 0

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal pending_new, pending_counts, rows_since_commit
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
                    [
                        next_source_row,
                        complete,
                        str(self.paths.game_file),
                        self.paths.dataset_id,
                    ],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            pending_new = []
            pending_counts = Counter()
            rows_since_commit = 0

        with open_csv_text(self.paths.game_file) as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != self.game_columns:
                raise ValueError("Game header changed between startup and order scan")
            for row in reader:
                if source_row < phase.next_source_row:
                    source_row += 1
                    continue
                draft_id = row[draft_position]
                if draft_id not in draft_to_index:
                    draft_index = next_index
                    draft_to_index[draft_id] = draft_index
                    next_index += 1
                    pending_new.append(
                        (
                            self.paths.dataset_id,
                            draft_index,
                            draft_id,
                            draft_index < checkpoint.next_draft_index,
                        )
                    )
                pending_counts[draft_id] += 1
                source_row += 1
                rows_since_commit += 1
                if rows_since_commit >= 50_000:
                    commit_progress(source_row)

        commit_progress(source_row, complete=True)
        LOG.info("Indexed %d drafts from game_data in one pass.", len(draft_to_index))

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
            LOG.info("Draft source phase already complete.")
            return

        target_draft_ids = self.incomplete_draft_ids(con)
        if not target_draft_ids:
            con.execute(
                "UPDATE source_ingestion_checkpoints SET completed = TRUE, updated_at = current_timestamp "
                "WHERE dataset_id = ? AND phase = 'draft'",
                [self.paths.dataset_id],
            )
            return

        # The old interleaved builder preloaded picks for an entire outer batch before
        # advancing next_draft_index.  On first use of the single-pass builder, remove
        # only those incomplete picks; fully checkpointed drafts are untouched.
        if phase.next_source_row == 0:
            con.execute("BEGIN")
            try:
                clear_partial_draft_picks(con, target_draft_ids)
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise

        index = {column: position for position, column in enumerate(self.draft_columns)}
        pack_fields = [
            (index[column], name, self.reference.card_name_to_id[name])
            for column, name in self.draft_pack_columns.items()
        ]
        pool_fields = [
            (index[column], self.reference.card_name_to_id[name])
            for column, name in self.draft_pool_columns.items()
        ]

        def value(row: list[str], column: str) -> str | None:
            position = index.get(column)
            return None if position is None else row[position]

        draft_rows: dict[str, dict[str, Any]] = {}
        pick_rows: list[dict[str, Any]] = []
        pick_card_rows: list[dict[str, Any]] = []
        source_row = 0
        raw_since_commit = 0
        total_picks = 0
        commit_pick_limit = max(500, self.batch_size * 5)

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal total_picks, raw_since_commit
            if not pick_rows and not complete and raw_since_commit < 50_000:
                return
            con.execute("BEGIN")
            try:
                if draft_rows:
                    upsert_draft_metadata(con, list(draft_rows.values()))
                if pick_rows:
                    insert_rows(
                        con,
                        "draft_picks",
                        pick_rows,
                        int64=("pick_id", "card_id"),
                        int16=("pack_number", "pick_number"),
                    )
                    insert_rows(
                        con,
                        "draft_pick_cards",
                        pick_card_rows,
                        int64=("pick_id", "card_id"),
                        int16=("pack_count", "pool_count"),
                    )
                con.execute(
                    """
                    UPDATE source_ingestion_checkpoints
                    SET next_source_row = ?, completed = ?, source_file = ?,
                        updated_at = current_timestamp
                    WHERE dataset_id = ? AND phase = 'draft'
                    """,
                    [
                        next_source_row,
                        complete,
                        str(self.paths.draft_file),
                        self.paths.dataset_id,
                    ],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            total_picks += len(pick_rows)
            draft_rows.clear()
            pick_rows.clear()
            pick_card_rows.clear()
            raw_since_commit = 0

        with open_csv_text(self.paths.draft_file) as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != self.draft_columns:
                raise ValueError("Draft header changed between startup and source scan")
            for row in reader:
                if source_row < phase.next_source_row:
                    source_row += 1
                    continue
                source_row += 1
                raw_since_commit += 1
                draft_id = row[index["draft_id"]]
                if draft_id not in target_draft_ids:
                    if raw_since_commit >= 50_000:
                        commit_progress(source_row)
                    continue

                metadata = {
                    "draft_id": draft_id,
                    "expansion": text_or_none(value(row, "expansion")),
                    "event_type": text_or_none(value(row, "event_type")),
                    "draft_time": timestamp_or_none(value(row, "draft_time")),
                    "rank": text_or_none(value(row, "rank")),
                    "event_match_wins": integer_or_none(value(row, "event_match_wins")),
                    "event_match_losses": integer_or_none(value(row, "event_match_losses")),
                    "user_n_games_bucket": integer_or_none(value(row, "user_n_games_bucket")),
                    "user_game_win_rate_bucket": numeric_or_none(value(row, "user_game_win_rate_bucket")),
                }
                previous = draft_rows.get(draft_id)
                if previous is None:
                    draft_rows[draft_id] = metadata
                else:
                    for field, incoming in metadata.items():
                        if field == "draft_id" or incoming is None:
                            continue
                        stored = previous.get(field)
                        if stored is None:
                            previous[field] = incoming
                        elif not values_equal((stored,), (incoming,)):
                            raise ValueError(f"Conflicting {field} within draft {draft_id}")

                pack_number = parse_key_integer(row[index["pack_number"]])
                pick_number = parse_key_integer(row[index["pick_number"]])
                picked_name = normalize_card_name(row[index["pick"]])
                picked_id = self.reference.card_name_to_id[picked_name]
                pick_id = make_pick_id(draft_id, pack_number, pick_number)

                card_counts: dict[int, list[int]] = {}
                picked_offered = False
                for position, name, card_id in pack_fields:
                    count = source_count(row[position])
                    if count:
                        card_counts.setdefault(card_id, [0, 0])[0] = count
                        if name == picked_name:
                            picked_offered = True
                if not picked_offered:
                    raise ValueError(
                        f"Selected card {picked_name!r} not offered for {draft_id} "
                        f"pack={pack_number} pick={pick_number}"
                    )
                for position, card_id in pool_fields:
                    count = source_count(row[position])
                    if count:
                        card_counts.setdefault(card_id, [0, 0])[1] = count

                pick_rows.append({
                    "pick_id": pick_id,
                    "draft_id": draft_id,
                    "pack_number": pack_number,
                    "pick_number": pick_number,
                    "card_id": picked_id,
                    "pick_maindeck_rate": numeric_or_none(value(row, "pick_maindeck_rate")),
                    "pick_sideboard_in_rate": numeric_or_none(value(row, "pick_sideboard_in_rate")),
                })
                pick_card_rows.extend(
                    {
                        "pick_id": pick_id,
                        "card_id": card_id,
                        "pack_count": counts[0],
                        "pool_count": counts[1],
                    }
                    for card_id, counts in card_counts.items()
                )

                if len(pick_rows) >= commit_pick_limit:
                    commit_progress(source_row)

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
        LOG.info("Draft source pass complete: imported %d picks.", total_picks)

    def build_game_player_rows_without_replay(
        self,
        game_batch: pd.DataFrame,
        game_ids: list[int],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for position, game in enumerate(game_batch.to_dict("records")):
            game_id = game_ids[position]
            rows.append({
                "game_id": game_id,
                "is_user": True,
                "rank": text_or_none(game.get("rank")),
                "main_colors": text_or_none(game.get("main_colors")),
                "splash_colors": text_or_none(game.get("splash_colors")),
                "observed_colors": None,
                "num_mulligans": integer_or_none(game.get("num_mulligans")),
                "n_games_bucket": None,
                "game_win_rate_bucket": None,
            })
            rows.append({
                "game_id": game_id,
                "is_user": False,
                "rank": text_or_none(game.get("opp_rank")),
                "main_colors": None,
                "splash_colors": None,
                "observed_colors": text_or_none(game.get("opp_colors")),
                "num_mulligans": integer_or_none(game.get("opp_num_mulligans")),
                "n_games_bucket": None,
                "game_win_rate_bucket": None,
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
            LOG.info("Game source phase already complete.")
            return

        target_draft_ids = self.incomplete_draft_ids(con)
        if phase.next_source_row == 0:
            clear_partial_game_data(con, self.paths.dataset_id)

        wanted_columns = self.game_usecols()
        index = {column: position for position, column in enumerate(self.game_columns)}
        wanted_indices = [index[column] for column in wanted_columns]
        draft_position = index["draft_id"]
        rows: list[list[str]] = []
        source_row = 0
        raw_since_commit = 0
        total_games = 0
        chunk_limit = max(500, self.batch_size * 5)

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal total_games, raw_since_commit
            if not rows and not complete and raw_since_commit < 50_000:
                return
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
                    [
                        next_source_row,
                        complete,
                        str(self.paths.game_file),
                        self.paths.dataset_id,
                    ],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            total_games += len(rows)
            rows.clear()
            raw_since_commit = 0

        with open_csv_text(self.paths.game_file) as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != self.game_columns:
                raise ValueError("Game header changed between startup and source scan")
            for row in reader:
                if source_row < phase.next_source_row:
                    source_row += 1
                    continue
                source_row += 1
                raw_since_commit += 1
                if row[draft_position] in target_draft_ids:
                    rows.append([row[position] for position in wanted_indices])
                if len(rows) >= chunk_limit or raw_since_commit >= 50_000:
                    commit_progress(source_row)

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
        LOG.info("Game source pass complete: imported %d games.", total_games)

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
        for position, game_id in enumerate(game_ids):
            if game_id not in expected:
                raise ValueError(f"Replay row has no staged game row: game_id={game_id}")
            on_play, won, num_turns, user_mulls, opp_mulls = expected[game_id]
            checks = (
                ("on_play", parse_bool(replay_batch.iloc[position].get("on_play")), on_play),
                ("won", parse_bool(replay_batch.iloc[position].get("won")), won),
                ("num_turns", integer_or_none(replay_batch.iloc[position].get("num_turns")), num_turns),
                ("num_mulligans", integer_or_none(replay_batch.iloc[position].get("num_mulligans")), user_mulls),
                ("opp_num_mulligans", integer_or_none(replay_batch.iloc[position].get("opp_num_mulligans")), opp_mulls),
            )
            for field, source, stored in checks:
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
            LOG.info("Replay source phase already complete.")
            return

        target_draft_ids = self.incomplete_draft_ids(con)
        if phase.next_source_row == 0:
            clear_partial_replay_data(con, self.paths.dataset_id)

        wanted_columns = self.replay_usecols()
        index = {column: position for position, column in enumerate(self.replay_columns)}
        missing_columns = [column for column in wanted_columns if column not in index]
        if missing_columns:
            raise ValueError(f"Replay CSV missing requested columns: {missing_columns[:20]}")
        wanted_indices = [index[column] for column in wanted_columns]
        draft_position = index["draft_id"]
        rows: list[list[str]] = []
        source_row = 0
        raw_since_commit = 0
        total_replays = 0
        chunk_limit = max(50, self.batch_size)

        def commit_progress(next_source_row: int, complete: bool = False) -> None:
            nonlocal total_replays, raw_since_commit
            if not rows and not complete and raw_since_commit < 50_000:
                return
            replay_batch = pd.DataFrame(rows, columns=wanted_columns) if rows else None
            con.execute("BEGIN")
            try:
                if replay_batch is not None and len(replay_batch):
                    game_ids = make_game_ids(replay_batch)
                    if len(set(game_ids)) != len(game_ids):
                        raise ValueError("Duplicate natural game key inside replay source chunk")
                    self.validate_replay_chunk_against_database(con, replay_batch, game_ids)

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
                    self.update_replay_user_metrics(con, replay_batch, game_ids)

                    counts = Counter(replay_batch["draft_id"].astype(str))
                    con.executemany(
                        """
                        UPDATE ingestion_draft_status
                        SET replay_games_processed = replay_games_processed + ?
                        WHERE dataset_id = ? AND draft_id = ?
                        """,
                        [
                            (count, self.paths.dataset_id, draft_id)
                            for draft_id, count in counts.items()
                        ],
                    )
                    overflow = con.execute(
                        """
                        SELECT draft_id, replay_games_processed, expected_games
                        FROM ingestion_draft_status
                        WHERE dataset_id = ? AND replay_games_processed > expected_games
                        LIMIT 10
                        """,
                        [self.paths.dataset_id],
                    ).fetchall()
                    if overflow:
                        raise ValueError(f"Replay source has more games than game_data: {overflow}")
                    con.execute(
                        """
                        UPDATE ingestion_draft_status
                        SET completed = (replay_games_processed = expected_games)
                        WHERE dataset_id = ? AND completed = FALSE
                        """,
                        [self.paths.dataset_id],
                    )
                    self.advance_main_checkpoint_from_status(con, total_drafts)

                con.execute(
                    """
                    UPDATE source_ingestion_checkpoints
                    SET next_source_row = ?, completed = ?, source_file = ?,
                        updated_at = current_timestamp
                    WHERE dataset_id = ? AND phase = 'replay'
                    """,
                    [
                        next_source_row,
                        complete,
                        str(self.paths.replay_file),
                        self.paths.dataset_id,
                    ],
                )
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            total_replays += len(rows)
            rows.clear()
            raw_since_commit = 0

        with open_csv_text(self.paths.replay_file) as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header != self.replay_columns:
                raise ValueError("Replay header changed between startup and source scan")
            for row in reader:
                if source_row < phase.next_source_row:
                    source_row += 1
                    continue
                source_row += 1
                raw_since_commit += 1
                if len(row) != len(self.replay_columns):
                    raise ValueError(
                        f"Malformed replay CSV row {source_row}: {len(row)} fields, "
                        f"expected {len(self.replay_columns)}"
                    )
                if row[draft_position] in target_draft_ids:
                    rows.append([row[position] for position in wanted_indices])
                if len(rows) >= chunk_limit or raw_since_commit >= 50_000:
                    commit_progress(source_row)
                if self.replay_progress_every > 0 and source_row % self.replay_progress_every == 0:
                    LOG.info("Replay source progress: %d rows scanned", source_row)

        commit_progress(source_row, complete=True)

        missing = con.execute(
            """
            SELECT draft_id, replay_games_processed, expected_games
            FROM ingestion_draft_status
            WHERE dataset_id = ? AND completed = FALSE
            LIMIT 10
            """,
            [self.paths.dataset_id],
        ).fetchall()
        if missing:
            raise ValueError(f"Replay source completed before all games were found: {missing}")
        con.execute("CHECKPOINT")
        LOG.info("Replay source pass complete: imported %d replay rows.", total_replays)

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
        "--batch-size",
        type=int,
        default=100,
        help="Base transaction chunk size. Source files are scanned once regardless of this value.",
    )
    parser.add_argument(
        "--replay-progress-every",
        type=int,
        default=500_000,
        help="Log replay-scan progress every N rows; use 0 to disable.",
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
            batch_size=args.batch_size,
            overwrite=args.overwrite and index == 1,
            replay_progress_every=args.replay_progress_every,
        ).run()


if __name__ == "__main__":
    main()
