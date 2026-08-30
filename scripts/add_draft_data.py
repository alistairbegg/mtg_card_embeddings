#!/usr/bin/env python3
"""Add 17Lands public draft-pick data to an existing normalized DuckDB.

By default this script discovers draft_data_public.*.csv.gz files in --data-dir and
imports only draft_id values that already exist in the database. This makes it a
safe backfill for a database that was originally built from game/replay data.

The importer is resumable and idempotent. If the source file or the allowed set
of draft IDs changes, it rescans the source and upserts the affected draft rows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
import numpy as np
import pandas as pd


LOG = logging.getLogger("17lands-drafts")

DRAFT_REQUIRED_COLUMNS = {
    "expansion",
    "event_type",
    "draft_id",
    "draft_time",
    "pack_number",
    "pick_number",
    "pick",
}

DRAFT_OPTIONAL_COLUMNS = {
    "rank",
    "event_match_wins",
    "event_match_losses",
    "pick_2",
    "pick_maindeck_rate",
    "pick_sideboard_in_rate",
    "user_n_games_bucket",
    "user_game_win_rate_bucket",
}

PACK_PREFIX = "pack_card_"
POOL_PREFIX = "pool_"

DRAFT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS draft_details (
    draft_id VARCHAR PRIMARY KEY REFERENCES drafts(draft_id),
    rank VARCHAR,
    event_match_wins SMALLINT,
    event_match_losses SMALLINT,
    user_n_games_bucket SMALLINT,
    user_game_win_rate_bucket REAL
);

CREATE TABLE IF NOT EXISTS draft_picks (
    draft_pick_id BIGINT PRIMARY KEY,
    draft_id VARCHAR NOT NULL REFERENCES drafts(draft_id),
    source_pack_number SMALLINT NOT NULL,
    source_pick_number SMALLINT NOT NULL,
    pick_maindeck_rate REAL,
    pick_sideboard_in_rate REAL,
    UNIQUE (draft_id, source_pack_number, source_pick_number)
);

CREATE TABLE IF NOT EXISTS draft_pick_selections (
    draft_pick_id BIGINT NOT NULL REFERENCES draft_picks(draft_pick_id),
    selection_number SMALLINT NOT NULL,
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    PRIMARY KEY (draft_pick_id, selection_number)
);

CREATE TABLE IF NOT EXISTS draft_pick_options (
    draft_pick_id BIGINT NOT NULL REFERENCES draft_picks(draft_pick_id),
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    quantity SMALLINT NOT NULL,
    PRIMARY KEY (draft_pick_id, card_id)
);

CREATE TABLE IF NOT EXISTS draft_pick_pool_cards (
    draft_pick_id BIGINT NOT NULL REFERENCES draft_picks(draft_pick_id),
    card_id BIGINT NOT NULL REFERENCES cards(card_id),
    quantity SMALLINT NOT NULL,
    PRIMARY KEY (draft_pick_id, card_id)
);

CREATE TABLE IF NOT EXISTS draft_ingestion_checkpoints (
    dataset_id VARCHAR PRIMARY KEY,
    draft_file VARCHAR NOT NULL,
    draft_signature VARCHAR NOT NULL,
    filter_signature VARCHAR NOT NULL,
    next_draft_row BIGINT NOT NULL,
    committed_batches BIGINT NOT NULL,
    completed BOOLEAN NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
"""


@dataclass(frozen=True)
class DraftCheckpoint:
    next_draft_row: int
    committed_batches: int
    completed: bool


def deterministic_bigint(*parts: Any) -> int:
    value = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def normalize_card_name(name: Any) -> str:
    return str(name).strip()


def make_card_id(card_name: str) -> int:
    return deterministic_bigint("card", normalize_card_name(card_name))


def make_draft_pick_id(draft_id: str, pack_number: int, pick_number: int) -> int:
    return deterministic_bigint("draft_pick", draft_id, pack_number, pick_number)


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


def text_or_none(value: Any) -> str | None:
    if is_missing_or_blank(value):
        return None
    text = str(value).strip()
    return text or None


def numeric_or_none(value: Any) -> float | None:
    if is_missing_or_blank(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(numeric) else numeric


def integer_or_none(value: Any) -> int | None:
    numeric = numeric_or_none(value)
    if numeric is None:
        return None
    if not float(numeric).is_integer():
        raise ValueError(f"Expected integer, got {value!r}")
    return int(numeric)


def timestamp_or_none(value: Any) -> pd.Timestamp | None:
    if is_missing_or_blank(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed


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
    con.execute(DRAFT_SCHEMA_SQL)


def sync_draft_card_names(
    con: duckdb.DuckDBPyConnection,
    card_names: Iterable[str],
) -> dict[str, int]:
    names = sorted({normalize_card_name(name) for name in card_names if normalize_card_name(name)})
    mapping = {name: make_card_id(name) for name in names}

    existing_by_name = {
        str(name): int(card_id)
        for card_id, name in con.execute("SELECT card_id, card_name FROM cards").fetchall()
    }
    existing_by_id = {
        int(card_id): str(name)
        for card_id, name in con.execute("SELECT card_id, card_name FROM cards").fetchall()
    }

    new_rows = []
    for name, card_id in mapping.items():
        existing_id = existing_by_name.get(name)
        if existing_id is not None:
            if existing_id != card_id:
                raise ValueError(f"Canonical card ID mismatch for {name!r}")
            continue
        conflicting_name = existing_by_id.get(card_id)
        if conflicting_name is not None and conflicting_name != name:
            raise ValueError(
                f"Deterministic card ID collision: {card_id} is both {conflicting_name!r} and {name!r}"
            )
        new_rows.append((card_id, name))

    if new_rows:
        con.executemany("INSERT INTO cards (card_id, card_name) VALUES (?, ?)", new_rows)
        LOG.info("Draft reference sync: +%d card names", len(new_rows))

    return mapping


def iter_source_batches(
    draft_file: Path,
    usecols: list[str],
    chunk_size: int,
    start_row: int,
) -> Iterator[tuple[int, pd.DataFrame, int]]:
    current_row = 0
    for chunk in pd.read_csv(
        draft_file,
        usecols=usecols,
        dtype="string",
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk_start = current_row
        raw_len = len(chunk)
        chunk_end = chunk_start + raw_len
        current_row = chunk_end

        if chunk_end <= start_row:
            continue
        if start_row > chunk_start:
            offset = start_row - chunk_start
            chunk = chunk.iloc[offset:].copy()
            chunk_start = start_row

        if len(chunk):
            yield chunk_start, chunk.reset_index(drop=True), chunk_end


def load_or_reset_checkpoint(
    con: duckdb.DuckDBPyConnection,
    *,
    dataset_id: str,
    draft_file: Path,
    draft_signature: str,
    filter_signature: str,
) -> DraftCheckpoint:
    row = con.execute(
        """
        SELECT draft_signature, filter_signature, next_draft_row,
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
                next_draft_row, committed_batches, completed, updated_at
            ) VALUES (?, ?, ?, ?, 0, 0, FALSE, current_timestamp)
            """,
            [dataset_id, str(draft_file), draft_signature, filter_signature],
        )
        con.execute("CHECKPOINT")
        return DraftCheckpoint(0, 0, False)

    saved_draft_signature, saved_filter_signature, next_row, batches, completed = row
    if saved_draft_signature == draft_signature and saved_filter_signature == filter_signature:
        return DraftCheckpoint(int(next_row), int(batches), bool(completed))

    LOG.info(
        "Draft source/filter changed for %s; rescanning from row 0 with idempotent upserts.",
        dataset_id,
    )
    con.execute(
        """
        UPDATE draft_ingestion_checkpoints
        SET draft_file = ?, draft_signature = ?, filter_signature = ?,
            next_draft_row = 0, committed_batches = 0, completed = FALSE,
            updated_at = current_timestamp
        WHERE dataset_id = ?
        """,
        [str(draft_file), draft_signature, filter_signature, dataset_id],
    )
    con.execute("CHECKPOINT")
    return DraftCheckpoint(0, 0, False)


def _merge_detail_value(current: Any, incoming: Any, *, field: str, draft_id: str) -> Any:
    if incoming is None:
        return current
    if current is None:
        return incoming
    if current != incoming:
        raise ValueError(f"Conflicting {field} values within draft {draft_id}: {current!r} != {incoming!r}")
    return current


def build_draft_rows(
    frame: pd.DataFrame,
    *,
    card_name_to_id: dict[str, int],
    pack_columns: dict[str, str],
    pool_columns: dict[str, str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    drafts: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, Any]] = {}
    picks: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    options: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []

    pack_matrix = frame[list(pack_columns)].apply(pd.to_numeric, errors="coerce").fillna(0)
    pool_matrix = frame[list(pool_columns)].apply(pd.to_numeric, errors="coerce").fillna(0)

    if (pack_matrix < 0).any().any() or (pool_matrix < 0).any().any():
        raise ValueError("Negative draft pack/pool card count found")

    for position, record in enumerate(frame.to_dict("records")):
        draft_id = str(record["draft_id"])
        pack_number = integer_or_none(record.get("pack_number"))
        pick_number = integer_or_none(record.get("pick_number"))
        if pack_number is None or pick_number is None:
            raise ValueError(f"Missing pack/pick number for draft {draft_id}")

        draft_row = {
            "draft_id": draft_id,
            "expansion": text_or_none(record.get("expansion")),
            "event_type": text_or_none(record.get("event_type")),
            "draft_time": timestamp_or_none(record.get("draft_time")),
        }
        previous_draft = drafts.get(draft_id)
        if previous_draft is not None and previous_draft != draft_row:
            raise ValueError(f"Conflicting draft metadata within source chunk for {draft_id}")
        drafts[draft_id] = draft_row

        incoming_detail = {
            "rank": text_or_none(record.get("rank")),
            "event_match_wins": integer_or_none(record.get("event_match_wins")),
            "event_match_losses": integer_or_none(record.get("event_match_losses")),
            "user_n_games_bucket": integer_or_none(record.get("user_n_games_bucket")),
            "user_game_win_rate_bucket": numeric_or_none(record.get("user_game_win_rate_bucket")),
        }
        detail = details.setdefault(
            draft_id,
            {"draft_id": draft_id, **{key: None for key in incoming_detail}},
        )
        for field, incoming in incoming_detail.items():
            detail[field] = _merge_detail_value(detail[field], incoming, field=field, draft_id=draft_id)

        draft_pick_id = make_draft_pick_id(draft_id, pack_number, pick_number)
        picks.append(
            {
                "draft_pick_id": draft_pick_id,
                "draft_id": draft_id,
                "source_pack_number": pack_number,
                "source_pick_number": pick_number,
                "pick_maindeck_rate": numeric_or_none(record.get("pick_maindeck_rate")),
                "pick_sideboard_in_rate": numeric_or_none(record.get("pick_sideboard_in_rate")),
            }
        )

        for selection_number, source_column in ((1, "pick"), (2, "pick_2")):
            if source_column not in frame.columns:
                continue
            picked_name = text_or_none(record.get(source_column))
            if picked_name is None:
                continue
            picked_name = normalize_card_name(picked_name)
            card_id = card_name_to_id.get(picked_name)
            if card_id is None:
                card_id = make_card_id(picked_name)
                card_name_to_id[picked_name] = card_id
            selections.append(
                {
                    "draft_pick_id": draft_pick_id,
                    "selection_number": selection_number,
                    "card_id": card_id,
                }
            )

        for column, card_name in pack_columns.items():
            quantity = int(pack_matrix.iloc[position][column])
            if quantity:
                options.append(
                    {
                        "draft_pick_id": draft_pick_id,
                        "card_id": card_name_to_id[card_name],
                        "quantity": quantity,
                    }
                )

        for column, card_name in pool_columns.items():
            quantity = int(pool_matrix.iloc[position][column])
            if quantity:
                pool_rows.append(
                    {
                        "draft_pick_id": draft_pick_id,
                        "card_id": card_name_to_id[card_name],
                        "quantity": quantity,
                    }
                )

    return drafts, details, picks, selections, options, pool_rows


def ensure_draft_parents(
    con: duckdb.DuckDBPyConnection,
    drafts: dict[str, dict[str, Any]],
    *,
    existing_only: bool,
) -> None:
    for draft_id, row in drafts.items():
        existing = con.execute(
            "SELECT expansion, event_type, draft_time FROM drafts WHERE draft_id = ?",
            [draft_id],
        ).fetchone()
        expected = (row["expansion"], row["event_type"], row["draft_time"])

        if existing is None:
            if existing_only:
                raise ValueError(f"Draft {draft_id} is not present in the existing database")
            con.execute(
                "INSERT INTO drafts (draft_id, expansion, event_type, draft_time) VALUES (?, ?, ?, ?)",
                [draft_id, *expected],
            )
            continue

        for stored, source, field in zip(existing, expected, ("expansion", "event_type", "draft_time"), strict=True):
            if stored is None or source is None:
                continue
            if field == "draft_time":
                equal = pd.Timestamp(stored) == pd.Timestamp(source)
            else:
                equal = stored == source
            if not equal:
                raise ValueError(
                    f"Draft metadata changed for {draft_id}: {field} {stored!r} != {source!r}"
                )


def upsert_draft_details(con: duckdb.DuckDBPyConnection, details: dict[str, dict[str, Any]]) -> None:
    for row in details.values():
        con.execute(
            """
            INSERT INTO draft_details (
                draft_id, rank, event_match_wins, event_match_losses,
                user_n_games_bucket, user_game_win_rate_bucket
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (draft_id) DO UPDATE SET
                rank = COALESCE(excluded.rank, draft_details.rank),
                event_match_wins = COALESCE(excluded.event_match_wins, draft_details.event_match_wins),
                event_match_losses = COALESCE(excluded.event_match_losses, draft_details.event_match_losses),
                user_n_games_bucket = COALESCE(excluded.user_n_games_bucket, draft_details.user_n_games_bucket),
                user_game_win_rate_bucket = COALESCE(
                    excluded.user_game_win_rate_bucket,
                    draft_details.user_game_win_rate_bucket
                )
            """,
            [
                row["draft_id"],
                row["rank"],
                row["event_match_wins"],
                row["event_match_losses"],
                row["user_n_games_bucket"],
                row["user_game_win_rate_bucket"],
            ],
        )


def _frame(rows: list[dict[str, Any]], int64: Iterable[str] = (), int16: Iterable[str] = ()) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in int64:
        if column in frame.columns:
            frame[column] = pd.array([row.get(column) for row in rows], dtype="Int64")
    for column in int16:
        if column in frame.columns:
            frame[column] = pd.array([row.get(column) for row in rows], dtype="Int16")
    return frame


def _insert_frame(con: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    temp = f"_draft_insert_{table}"
    con.register(temp, frame)
    try:
        columns = list(frame.columns)
        quoted = ", ".join(f'"{column}"' for column in columns)
        con.execute(f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM {temp}")
    finally:
        con.unregister(temp)


def replace_pick_rows(
    con: duckdb.DuckDBPyConnection,
    picks: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    options: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
) -> None:
    if not picks:
        return

    pick_frame = _frame(
        picks,
        int64=("draft_pick_id",),
        int16=("source_pack_number", "source_pick_number"),
    )
    con.register("_draft_pick_upserts", pick_frame)
    try:
        # Replace the complete source row atomically. This keeps reruns and
        # refreshed 17Lands dumps idempotent without depending on MERGE syntax.
        con.execute(
            "DELETE FROM draft_pick_selections WHERE draft_pick_id IN (SELECT draft_pick_id FROM _draft_pick_upserts)"
        )
        con.execute(
            "DELETE FROM draft_pick_options WHERE draft_pick_id IN (SELECT draft_pick_id FROM _draft_pick_upserts)"
        )
        con.execute(
            "DELETE FROM draft_pick_pool_cards WHERE draft_pick_id IN (SELECT draft_pick_id FROM _draft_pick_upserts)"
        )
        con.execute(
            "DELETE FROM draft_picks WHERE draft_pick_id IN (SELECT draft_pick_id FROM _draft_pick_upserts)"
        )
        con.execute(
            """
            INSERT INTO draft_picks (
                draft_pick_id, draft_id, source_pack_number, source_pick_number,
                pick_maindeck_rate, pick_sideboard_in_rate
            )
            SELECT
                draft_pick_id, draft_id, source_pack_number, source_pick_number,
                pick_maindeck_rate, pick_sideboard_in_rate
            FROM _draft_pick_upserts
            """
        )
    finally:
        con.unregister("_draft_pick_upserts")

    _insert_frame(
        con,
        "draft_pick_selections",
        _frame(selections, int64=("draft_pick_id", "card_id"), int16=("selection_number",)),
    )
    _insert_frame(
        con,
        "draft_pick_options",
        _frame(options, int64=("draft_pick_id", "card_id"), int16=("quantity",)),
    )
    _insert_frame(
        con,
        "draft_pick_pool_cards",
        _frame(pool_rows, int64=("draft_pick_id", "card_id"), int16=("quantity",)),
    )


def ingest_draft_file(
    con: duckdb.DuckDBPyConnection,
    draft_file: Path,
    *,
    dataset_id: str,
    allowed_draft_ids: set[str],
    existing_only: bool,
    chunk_size: int = 2_000,
    max_batches: int | None = None,
    require_all: bool = True,
) -> bool:
    """Import one public 17Lands draft file.

    Returns True when the whole source file was processed, False when stopped by
    max_batches with a resumable checkpoint.
    """
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
    if not pack_columns:
        raise ValueError(f"{draft_file.name} has no {PACK_PREFIX}* columns")
    if not pool_columns:
        raise ValueError(f"{draft_file.name} has no {POOL_PREFIX}* columns")

    # Card names are encoded directly in the wide pack/pool column names.
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

    signature = file_signature(draft_file, header)
    allowed_signature = set_signature(allowed_draft_ids)
    checkpoint = load_or_reset_checkpoint(
        con,
        dataset_id=dataset_id,
        draft_file=draft_file,
        draft_signature=signature,
        filter_signature=allowed_signature,
    )

    if checkpoint.completed:
        LOG.info("Draft dataset %s is already complete.", dataset_id)
        return True

    LOG.info(
        "Draft dataset %s: resuming at source row %d after %d committed batches; %d draft IDs allowed.",
        dataset_id,
        checkpoint.next_draft_row,
        checkpoint.committed_batches,
        len(allowed_draft_ids),
    )

    processed_this_run = 0
    exhausted = True
    seen_this_run: set[str] = set()

    for source_start, source_chunk, source_end in iter_source_batches(
        draft_file,
        usecols,
        chunk_size,
        checkpoint.next_draft_row,
    ):
        if max_batches is not None and processed_this_run >= max_batches:
            exhausted = False
            break

        filtered = source_chunk[source_chunk["draft_id"].astype(str).isin(allowed_draft_ids)].copy()
        seen_this_run.update(filtered["draft_id"].dropna().astype(str))

        con.execute("BEGIN")
        try:
            if not filtered.empty:
                # If an unexpected picked name is not present in the wide columns,
                # add it deterministically before inserting the selection row.
                pick_names = set()
                for column in ("pick", "pick_2"):
                    if column in filtered.columns:
                        pick_names.update(
                            normalize_card_name(value)
                            for value in filtered[column].dropna().astype(str)
                            if normalize_card_name(value)
                        )
                missing_names = pick_names - set(card_name_to_id)
                if missing_names:
                    card_name_to_id.update(sync_draft_card_names(con, missing_names))

                drafts, details, picks, selections, options, pool_rows = build_draft_rows(
                    filtered,
                    card_name_to_id=card_name_to_id,
                    pack_columns=pack_columns,
                    pool_columns=pool_columns,
                )
                ensure_draft_parents(con, drafts, existing_only=existing_only)
                upsert_draft_details(con, details)
                replace_pick_rows(con, picks, selections, options, pool_rows)

            con.execute(
                """
                UPDATE draft_ingestion_checkpoints
                SET next_draft_row = ?, committed_batches = ?, draft_file = ?,
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

        con.execute("CHECKPOINT")
        processed_this_run += 1
        checkpoint = DraftCheckpoint(
            next_draft_row=source_end,
            committed_batches=checkpoint.committed_batches + 1,
            completed=False,
        )
        LOG.info(
            "Draft batch %d committed: source rows %d..%d; matched %d rows from %d drafts.",
            checkpoint.committed_batches,
            source_start,
            source_end - 1,
            len(filtered),
            len(set(filtered["draft_id"].astype(str))) if not filtered.empty else 0,
        )

    if not exhausted:
        LOG.info(
            "Stopped draft ingestion cleanly after %d new batches. Resume with the same command.",
            processed_this_run,
        )
        return False

    con.execute("BEGIN")
    try:
        con.execute(
            """
            UPDATE draft_ingestion_checkpoints
            SET completed = TRUE, updated_at = current_timestamp
            WHERE dataset_id = ?
            """,
            [dataset_id],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("CHECKPOINT")

    if require_all and allowed_draft_ids:
        existing_with_picks = {
            str(draft_id)
            for (draft_id,) in con.execute(
                "SELECT DISTINCT draft_id FROM draft_picks"
            ).fetchall()
        }
        missing_ids = sorted(allowed_draft_ids - existing_with_picks)
        if missing_ids:
            raise ValueError(
                f"Draft source {draft_file.name} completed but {len(missing_ids)} allowed draft IDs "
                f"still have no draft picks; first IDs: {missing_ids[:10]}"
            )

    invalid_selections = int(
        con.execute(
            """
            SELECT count(*)
            FROM draft_pick_selections s
            LEFT JOIN draft_pick_options o
              ON o.draft_pick_id = s.draft_pick_id
             AND o.card_id = s.card_id
            WHERE o.card_id IS NULL
            """
        ).fetchone()[0]
    )
    if invalid_selections:
        LOG.warning(
            "%d draft selections are not represented in the corresponding pack options.",
            invalid_selections,
        )

    LOG.info("Draft dataset %s is complete.", dataset_id)
    return True


def discover_draft_files(data_dir: Path, suffix: str | None) -> list[tuple[str, Path]]:
    files = sorted(data_dir.glob("draft_data_public.*.csv.gz"))
    prefix = "draft_data_public."
    ending = ".csv.gz"
    pairs = [(path.name[len(prefix):-len(ending)], path) for path in files]

    if suffix is not None:
        matches = [pair for pair in pairs if pair[0] == suffix]
        if not matches:
            raise ValueError(f"No draft dataset found for suffix {suffix!r} in {data_dir}")
        return matches
    if not pairs:
        raise ValueError(f"No draft_data_public.*.csv.gz files found in {data_dir}")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/17lands.duckdb"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset-suffix")
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Process at most this many source chunks per draft file, then stop cleanly.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail if some existing database draft IDs are absent from the supplied draft files.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db_path = args.db.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    draft_files = discover_draft_files(data_dir, args.dataset_suffix)

    with duckdb.connect(str(db_path)) as con:
        ensure_draft_schema(con)
        allowed_draft_ids = {
            str(draft_id)
            for (draft_id,) in con.execute("SELECT draft_id FROM drafts").fetchall()
        }
        if not allowed_draft_ids:
            raise ValueError("The database has no drafts to backfill")

        LOG.info("Existing database drafts eligible for backfill: %d", len(allowed_draft_ids))

        for dataset_id, draft_file in draft_files:
            LOG.info("Draft CSV: %s", draft_file)
            ingest_draft_file(
                con,
                draft_file,
                dataset_id=dataset_id,
                allowed_draft_ids=allowed_draft_ids,
                existing_only=True,
                chunk_size=args.chunk_size,
                max_batches=args.max_batches,
                # A single file normally covers only one dataset. Global coverage
                # is checked after all discovered files below.
                require_all=False,
            )

        if not args.allow_missing:
            with_picks = {
                str(draft_id)
                for (draft_id,) in con.execute("SELECT DISTINCT draft_id FROM draft_picks").fetchall()
            }
            missing = sorted(allowed_draft_ids - with_picks)
            if missing:
                raise ValueError(
                    f"After scanning all draft files, {len(missing)} existing database drafts still "
                    f"have no pick data; first IDs: {missing[:10]}. Use --allow-missing if expected."
                )

        LOG.info(
            "Draft backfill complete: %d picks, %d offered-card rows, %d pool-card rows.",
            int(con.execute("SELECT count(*) FROM draft_picks").fetchone()[0]),
            int(con.execute("SELECT count(*) FROM draft_pick_options").fetchone()[0]),
            int(con.execute("SELECT count(*) FROM draft_pick_pool_cards").fetchone()[0]),
        )


if __name__ == "__main__":
    main()
