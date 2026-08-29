from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------

DB_PATH = Path("data/17lands.duckdb")
OUTPUT_TABLE = "deck_ppmi_svd_embeddings"

DECK_SIZE = 40
CARDS_SEEN = 10
MIN_PAIR_PROB = 1e-5
EMBEDDING_COMPONENTS = 64
RANDOM_STATE = 42

EXCLUDE_BASICS = True
BASIC_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}

LOG_LEVEL = "INFO"


LOG = logging.getLogger("deck_embeddings")


def quoted_sql_strings(values: set[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in sorted(values))


def validate_identifier(value: str) -> str:
    if not value or not value.replace("_", "").isalnum() or value[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def build_basic_filter(exclude_basics: bool) -> str:
    if not exclude_basics:
        return ""
    return f"AND c.card_name NOT IN ({quoted_sql_strings(BASIC_NAMES)})"


def load_embedding_card_table(con: duckdb.DuckDBPyConnection, basic_filter: str) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT DISTINCT
            dbc.card_id,
            c.card_name
        FROM deck_build_cards dbc
        JOIN cards c
            ON c.card_id = dbc.card_id
        WHERE dbc.deck_count > 0
        {basic_filter}
        ORDER BY dbc.card_id
        """
    ).df()


def load_all_cards(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            card_id,
            card_name
        FROM cards
        ORDER BY card_id
        """
    ).df()


def load_ppmi_pairs(
    con: duckdb.DuckDBPyConnection,
    *,
    deck_size: int,
    cards_seen: int,
    min_pair_prob: float,
    basic_filter: str,
) -> pd.DataFrame:
    # Each build contributes the probability that both cards would be seen in a
    # sample of CARDS_SEEN cards. A draft contributes only its strongest build
    # for that pair, so sideboarding/build changes do not multiply one draft.
    query = f"""
    WITH
    build_cards AS (
        SELECT
            db.draft_id,
            db.build_id,
            dbc.card_id,
            dbc.deck_count
        FROM deck_builds db
        JOIN deck_build_cards dbc
            ON dbc.build_id = db.build_id
        JOIN cards c
            ON c.card_id = dbc.card_id
        WHERE dbc.deck_count > 0
        {basic_filter}
    ),
    card_build_prob AS (
        SELECT
            draft_id,
            build_id,
            card_id,
            CASE
                WHEN {deck_size} - deck_count >= {cards_seen}
                THEN 1.0 - EXP(
                    LGAMMA({deck_size} - deck_count + 1)
                    - LGAMMA({cards_seen} + 1)
                    - LGAMMA({deck_size} - deck_count - {cards_seen} + 1)
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA({deck_size} - {cards_seen} + 1)
                    )
                )
                ELSE 1.0
            END AS p_seen
        FROM build_cards
    ),
    build_pair_prob AS (
        SELECT
            a.draft_id,
            a.build_id,
            a.card_id AS card_a_id,
            b.card_id AS card_b_id,
            1.0
            - CASE
                WHEN {deck_size} - a.deck_count >= {cards_seen}
                THEN EXP(
                    LGAMMA({deck_size} - a.deck_count + 1)
                    - LGAMMA({cards_seen} + 1)
                    - LGAMMA({deck_size} - a.deck_count - {cards_seen} + 1)
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA({deck_size} - {cards_seen} + 1)
                    )
                )
                ELSE 0.0
              END
            - CASE
                WHEN {deck_size} - b.deck_count >= {cards_seen}
                THEN EXP(
                    LGAMMA({deck_size} - b.deck_count + 1)
                    - LGAMMA({cards_seen} + 1)
                    - LGAMMA({deck_size} - b.deck_count - {cards_seen} + 1)
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA({deck_size} - {cards_seen} + 1)
                    )
                )
                ELSE 0.0
              END
            + CASE
                WHEN {deck_size} - a.deck_count - b.deck_count >= {cards_seen}
                THEN EXP(
                    LGAMMA({deck_size} - a.deck_count - b.deck_count + 1)
                    - LGAMMA({cards_seen} + 1)
                    - LGAMMA({deck_size} - a.deck_count - b.deck_count - {cards_seen} + 1)
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA({deck_size} - {cards_seen} + 1)
                    )
                )
                ELSE 0.0
              END AS p_seen_together
        FROM build_cards a
        JOIN build_cards b
            ON a.build_id = b.build_id
           AND a.card_id < b.card_id
    ),
    draft_card_prob AS (
        SELECT
            draft_id,
            card_id,
            MAX(p_seen) AS p_seen
        FROM card_build_prob
        GROUP BY draft_id, card_id
    ),
    draft_pair_prob AS (
        SELECT
            draft_id,
            card_a_id,
            card_b_id,
            MAX(p_seen_together) AS p_seen_together
        FROM build_pair_prob
        GROUP BY draft_id, card_a_id, card_b_id
    ),
    draft_count AS (
        SELECT COUNT(DISTINCT draft_id)::DOUBLE AS n
        FROM deck_builds
    ),
    card_prob AS (
        SELECT
            dcp.card_id,
            SUM(dcp.p_seen) / dc.n AS p_card
        FROM draft_card_prob dcp
        CROSS JOIN draft_count dc
        GROUP BY dcp.card_id, dc.n
    ),
    pair_prob AS (
        SELECT
            dpp.card_a_id,
            dpp.card_b_id,
            SUM(dpp.p_seen_together) / dc.n AS p_ab
        FROM draft_pair_prob dpp
        CROSS JOIN draft_count dc
        GROUP BY dpp.card_a_id, dpp.card_b_id, dc.n
    )
    SELECT
        pp.card_a_id,
        pp.card_b_id,
        pp.p_ab,
        ap.p_card AS p_a,
        bp.p_card AS p_b,
        LN(pp.p_ab / (ap.p_card * bp.p_card)) AS pmi
    FROM pair_prob pp
    JOIN card_prob ap
        ON ap.card_id = pp.card_a_id
    JOIN card_prob bp
        ON bp.card_id = pp.card_b_id
    WHERE pp.p_ab > ?
      AND ap.p_card > 0
      AND bp.p_card > 0
    """

    pairs = con.execute(query, [min_pair_prob]).df()
    pairs["ppmi"] = pairs["pmi"].clip(lower=0)
    return pairs[np.isfinite(pairs["ppmi"]) & (pairs["ppmi"] > 0)].copy()


def build_ppmi_matrix(
    pairs: pd.DataFrame,
    card_ids: list[int],
) -> tuple[coo_matrix, dict[int, int]]:
    card_to_idx = {card_id: i for i, card_id in enumerate(card_ids)}

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for row in pairs.itertuples(index=False):
        i = card_to_idx.get(int(row.card_a_id))
        j = card_to_idx.get(int(row.card_b_id))
        if i is None or j is None:
            continue

        value = float(row.ppmi)
        rows.extend([i, j])
        cols.extend([j, i])
        values.extend([value, value])

    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(len(card_ids), len(card_ids)),
        dtype=np.float64,
    ).tocsr()

    return matrix, card_to_idx


def fit_embeddings(
    ppmi_matrix,
    requested_components: int,
    random_state: int,
) -> tuple[np.ndarray, TruncatedSVD, int]:
    if ppmi_matrix.shape[0] < 2 or ppmi_matrix.nnz == 0:
        raise ValueError("Not enough positive PPMI structure to fit an embedding")

    components = min(requested_components, ppmi_matrix.shape[1] - 1)
    if components < 1:
        raise ValueError("Embedding requires at least one SVD component")

    if components != requested_components:
        LOG.warning(
            "Reducing embedding dimensions from %d to %d because only %d cards are eligible.",
            requested_components,
            components,
            ppmi_matrix.shape[1],
        )

    model = TruncatedSVD(
        n_components=components,
        random_state=random_state,
    )
    vectors = model.fit_transform(ppmi_matrix)

    # Cosine similarity is the main geometry we use downstream, so persist unit
    # vectors rather than requiring every consumer to normalize them again.
    vectors = normalize(vectors)

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding contains non-finite values")

    return vectors, model, components


def build_output_rows(
    all_cards: pd.DataFrame,
    embedding_cards: pd.DataFrame,
    ppmi_matrix,
    vectors: np.ndarray,
    *,
    components: int,
    deck_size: int,
    cards_seen: int,
    min_pair_prob: float,
    exclude_basics: bool,
    random_state: int,
    explained_variance: float,
) -> pd.DataFrame:
    card_ids = [int(card_id) for card_id in embedding_cards["card_id"].tolist()]
    idx_by_card = {card_id: i for i, card_id in enumerate(card_ids)}
    row_nnz = np.diff(ppmi_matrix.indptr)

    embedding_by_card: dict[int, list[float]] = {}
    for card_id, idx in idx_by_card.items():
        # A card with no positive PPMI edges has no learned deck-context signal.
        if row_nnz[idx] == 0:
            continue
        vector = vectors[idx]
        if np.linalg.norm(vector) == 0:
            continue
        embedding_by_card[card_id] = [float(value) for value in vector]

    rows = all_cards[["card_id"]].copy()
    rows["embedding"] = [embedding_by_card.get(int(card_id)) for card_id in rows["card_id"]]
    rows["has_embedding"] = rows["embedding"].notna()
    rows["dimensions"] = components
    rows["deck_size"] = deck_size
    rows["cards_seen"] = cards_seen
    rows["min_pair_prob"] = min_pair_prob
    rows["exclude_basics"] = exclude_basics
    rows["random_state"] = random_state
    rows["explained_variance"] = explained_variance

    return rows


def write_embedding_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: pd.DataFrame,
    expected_dimensions: int,
) -> None:
    table_name = validate_identifier(table_name)
    con.register("embedding_rows", rows)

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(
            f"""
            CREATE TABLE {table_name} (
                card_id BIGINT PRIMARY KEY REFERENCES cards(card_id),
                embedding DOUBLE[],
                has_embedding BOOLEAN NOT NULL,
                dimensions INTEGER NOT NULL,
                deck_size INTEGER NOT NULL,
                cards_seen INTEGER NOT NULL,
                min_pair_prob DOUBLE NOT NULL,
                exclude_basics BOOLEAN NOT NULL,
                random_state INTEGER NOT NULL,
                explained_variance DOUBLE NOT NULL,
                generated_at TIMESTAMP NOT NULL DEFAULT current_timestamp
            )
            """
        )
        con.execute(
            f"""
            INSERT INTO {table_name} (
                card_id,
                embedding,
                has_embedding,
                dimensions,
                deck_size,
                cards_seen,
                min_pair_prob,
                exclude_basics,
                random_state,
                explained_variance
            )
            SELECT
                card_id,
                embedding,
                has_embedding,
                dimensions,
                deck_size,
                cards_seen,
                min_pair_prob,
                exclude_basics,
                random_state,
                explained_variance
            FROM embedding_rows
            """
        )

        total_cards = int(con.execute("SELECT count(*) FROM cards").fetchone()[0])
        stored_cards = int(con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0])
        duplicate_cards = int(
            con.execute(
                f"""
                SELECT count(*)
                FROM (
                    SELECT card_id
                    FROM {table_name}
                    GROUP BY card_id
                    HAVING count(*) <> 1
                )
                """
            ).fetchone()[0]
        )
        wrong_dimensions = int(
            con.execute(
                f"""
                SELECT count(*)
                FROM {table_name}
                WHERE embedding IS NOT NULL
                  AND len(embedding) <> ?
                """,
                [expected_dimensions],
            ).fetchone()[0]
        )
        flag_mismatches = int(
            con.execute(
                f"""
                SELECT count(*)
                FROM {table_name}
                WHERE has_embedding <> (embedding IS NOT NULL)
                """
            ).fetchone()[0]
        )

        if stored_cards != total_cards:
            raise ValueError(
                f"Embedding table has {stored_cards} rows but cards has {total_cards} rows"
            )
        if duplicate_cards:
            raise ValueError(f"Embedding table has {duplicate_cards} duplicate card IDs")
        if wrong_dimensions:
            raise ValueError(f"Embedding table has {wrong_dimensions} vectors with the wrong size")
        if flag_mismatches:
            raise ValueError(f"Embedding table has {flag_mismatches} has_embedding mismatches")

        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("embedding_rows")


def generate_embeddings(
    *,
    db_path: Path,
    output_table: str,
    deck_size: int,
    cards_seen: int,
    min_pair_prob: float,
    embedding_components: int,
    random_state: int,
    exclude_basics: bool,
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    if deck_size <= 0:
        raise ValueError("deck_size must be positive")
    if not 0 < cards_seen <= deck_size:
        raise ValueError("cards_seen must be between 1 and deck_size")
    if min_pair_prob < 0:
        raise ValueError("min_pair_prob cannot be negative")
    if embedding_components < 1:
        raise ValueError("embedding_components must be positive")

    output_table = validate_identifier(output_table)
    basic_filter = build_basic_filter(exclude_basics)

    LOG.info("Database: %s", db_path.resolve())
    LOG.info("Output table: %s", output_table)
    LOG.info(
        "Hyperparameters: deck_size=%d cards_seen=%d min_pair_prob=%g dimensions=%d exclude_basics=%s",
        deck_size,
        cards_seen,
        min_pair_prob,
        embedding_components,
        exclude_basics,
    )

    with duckdb.connect(str(db_path)) as con:
        all_cards = load_all_cards(con)
        embedding_cards = load_embedding_card_table(con, basic_filter)

        if embedding_cards.empty:
            raise ValueError("No maindeck cards were found in deck_build_cards")

        LOG.info(
            "Cards: %d total reference cards; %d eligible decklist cards.",
            len(all_cards),
            len(embedding_cards),
        )

        pairs = load_ppmi_pairs(
            con,
            deck_size=deck_size,
            cards_seen=cards_seen,
            min_pair_prob=min_pair_prob,
            basic_filter=basic_filter,
        )
        LOG.info("Positive-PMI pairs: %d", len(pairs))

        ppmi_matrix, _ = build_ppmi_matrix(
            pairs,
            [int(card_id) for card_id in embedding_cards["card_id"].tolist()],
        )
        LOG.info(
            "PPMI matrix: %d x %d with %d non-zero entries.",
            ppmi_matrix.shape[0],
            ppmi_matrix.shape[1],
            ppmi_matrix.nnz,
        )

        vectors, model, components = fit_embeddings(
            ppmi_matrix,
            embedding_components,
            random_state,
        )
        explained_variance = float(model.explained_variance_ratio_.sum())
        LOG.info("SVD explained variance: %.4f", explained_variance)

        rows = build_output_rows(
            all_cards,
            embedding_cards,
            ppmi_matrix,
            vectors,
            components=components,
            deck_size=deck_size,
            cards_seen=cards_seen,
            min_pair_prob=min_pair_prob,
            exclude_basics=exclude_basics,
            random_state=random_state,
            explained_variance=explained_variance,
        )

        embedded = int(rows["has_embedding"].sum())
        LOG.info(
            "Persisting embeddings for %d cards; %d cards will have NULL embeddings.",
            embedded,
            len(rows) - embedded,
        )

        write_embedding_table(
            con,
            output_table,
            rows,
            expected_dimensions=components,
        )

        # These are deliberately small post-write checks: row coverage, vector
        # dimensionality, and referential integrity are enforced before commit.
        stored = int(con.execute(f"SELECT count(*) FROM {output_table}").fetchone()[0])
        stored_embeddings = int(
            con.execute(
                f"SELECT count(*) FROM {output_table} WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        )
        LOG.info(
            "Done: %d table rows, %d learned embeddings, %d dimensions.",
            stored,
            stored_embeddings,
            components,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deck-context PPMI/SVD card embeddings and store them in DuckDB."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output-table", default=OUTPUT_TABLE)
    parser.add_argument("--deck-size", type=int, default=DECK_SIZE)
    parser.add_argument("--cards-seen", type=int, default=CARDS_SEEN)
    parser.add_argument("--min-pair-prob", type=float, default=MIN_PAIR_PROB)
    parser.add_argument("--dimensions", type=int, default=EMBEDDING_COMPONENTS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--include-basics",
        action="store_true",
        help="Include basic lands in the PPMI matrix. The default excludes them.",
    )
    parser.add_argument("--log-level", default=LOG_LEVEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    generate_embeddings(
        db_path=args.db,
        output_table=args.output_table,
        deck_size=args.deck_size,
        cards_seen=args.cards_seen,
        min_pair_prob=args.min_pair_prob,
        embedding_components=args.dimensions,
        random_state=args.random_state,
        exclude_basics=not args.include_basics,
    )


if __name__ == "__main__":
    main()
