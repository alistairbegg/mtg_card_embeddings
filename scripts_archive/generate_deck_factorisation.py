from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------

DB_PATH = Path("data/17lands.duckdb")

ARTIFACT_DIR = Path("ml_artifacts")
ARTIFACT_FILE = "deck_ppmi_svd.npz"

DECK_SIZE = 40
CARDS_SEEN = 10
MIN_PAIR_PROB = 1e-5
SVD_COMPONENTS = 64
RANDOM_STATE = 42

EXCLUDE_BASICS = True
BASIC_NAMES = {
    "Plains",
    "Island",
    "Swamp",
    "Mountain",
    "Forest",
    "Wastes",
}

LOG_LEVEL = "INFO"


LOG = logging.getLogger("deck_ppmi_svd")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def quoted_sql_strings(values: set[str]) -> str:
    return ", ".join(
        "'" + value.replace("'", "''") + "'"
        for value in sorted(values)
    )


def build_basic_filter(exclude_basics: bool) -> str:
    if not exclude_basics:
        return ""

    return (
        f"AND c.card_name NOT IN "
        f"({quoted_sql_strings(BASIC_NAMES)})"
    )


# -----------------------------------------------------------------------------
# Card universe
# -----------------------------------------------------------------------------

def load_embedding_card_table(
    con: duckdb.DuckDBPyConnection,
    basic_filter: str,
) -> pd.DataFrame:
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


# -----------------------------------------------------------------------------
# PPMI
# -----------------------------------------------------------------------------

def load_ppmi_pairs(
    con: duckdb.DuckDBPyConnection,
    *,
    deck_size: int,
    cards_seen: int,
    min_pair_prob: float,
    basic_filter: str,
) -> pd.DataFrame:
    """
    Calculate probability-weighted positive PMI for card pairs.

    Each build contributes the probability that both cards would be
    observed in a sample of `cards_seen` cards.

    A draft contributes only its strongest build for a card or pair,
    preventing sideboarding/build changes from counting the same draft
    multiple times.
    """

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
                    - LGAMMA(
                        {deck_size}
                        - deck_count
                        - {cards_seen}
                        + 1
                    )
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA(
                            {deck_size}
                            - {cards_seen}
                            + 1
                        )
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
                    LGAMMA(
                        {deck_size}
                        - a.deck_count
                        + 1
                    )
                    - LGAMMA({cards_seen} + 1)
                    - LGAMMA(
                        {deck_size}
                        - a.deck_count
                        - {cards_seen}
                        + 1
                    )
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA(
                            {deck_size}
                            - {cards_seen}
                            + 1
                        )
                    )
                )
                ELSE 0.0
            END

            - CASE
                WHEN {deck_size} - b.deck_count >= {cards_seen}
                THEN EXP(
                    LGAMMA(
                        {deck_size}
                        - b.deck_count
                        + 1
                    )
                    - LGAMMA({cards_seen} + 1)
                    - LGAMMA(
                        {deck_size}
                        - b.deck_count
                        - {cards_seen}
                        + 1
                    )
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA(
                            {deck_size}
                            - {cards_seen}
                            + 1
                        )
                    )
                )
                ELSE 0.0
            END

            + CASE
                WHEN (
                    {deck_size}
                    - a.deck_count
                    - b.deck_count
                ) >= {cards_seen}
                THEN EXP(
                    LGAMMA(
                        {deck_size}
                        - a.deck_count
                        - b.deck_count
                        + 1
                    )
                    - LGAMMA({cards_seen} + 1)
                    - LGAMMA(
                        {deck_size}
                        - a.deck_count
                        - b.deck_count
                        - {cards_seen}
                        + 1
                    )
                    - (
                        LGAMMA({deck_size} + 1)
                        - LGAMMA({cards_seen} + 1)
                        - LGAMMA(
                            {deck_size}
                            - {cards_seen}
                            + 1
                        )
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
        GROUP BY
            draft_id,
            card_id
    ),

    draft_pair_prob AS (
        SELECT
            draft_id,
            card_a_id,
            card_b_id,
            MAX(p_seen_together) AS p_seen_together
        FROM build_pair_prob
        GROUP BY
            draft_id,
            card_a_id,
            card_b_id
    ),

    draft_count AS (
        SELECT
            COUNT(DISTINCT draft_id)::DOUBLE AS n
        FROM deck_builds
    ),

    card_prob AS (
        SELECT
            dcp.card_id,
            SUM(dcp.p_seen) / dc.n AS p_card
        FROM draft_card_prob dcp
        CROSS JOIN draft_count dc
        GROUP BY
            dcp.card_id,
            dc.n
    ),

    pair_prob AS (
        SELECT
            dpp.card_a_id,
            dpp.card_b_id,
            SUM(dpp.p_seen_together) / dc.n AS p_ab
        FROM draft_pair_prob dpp
        CROSS JOIN draft_count dc
        GROUP BY
            dpp.card_a_id,
            dpp.card_b_id,
            dc.n
    )

    SELECT
        pp.card_a_id,
        pp.card_b_id,
        pp.p_ab,
        ap.p_card AS p_a,
        bp.p_card AS p_b,
        LN(
            pp.p_ab / (ap.p_card * bp.p_card)
        ) AS pmi

    FROM pair_prob pp

    JOIN card_prob ap
        ON ap.card_id = pp.card_a_id

    JOIN card_prob bp
        ON bp.card_id = pp.card_b_id

    WHERE pp.p_ab > ?
      AND ap.p_card > 0
      AND bp.p_card > 0
    """

    pairs = con.execute(
        query,
        [min_pair_prob],
    ).df()

    pairs["ppmi"] = pairs["pmi"].clip(
        lower=0,
    )

    return pairs[
        np.isfinite(pairs["ppmi"])
        & (pairs["ppmi"] > 0)
    ].copy()


# -----------------------------------------------------------------------------
# Sparse PPMI matrix
# -----------------------------------------------------------------------------

def build_ppmi_matrix(
    pairs: pd.DataFrame,
    card_ids: list[int],
):
    card_to_idx = {
        card_id: i
        for i, card_id in enumerate(card_ids)
    }

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for row in pairs.itertuples(index=False):
        i = card_to_idx.get(
            int(row.card_a_id)
        )
        j = card_to_idx.get(
            int(row.card_b_id)
        )

        if i is None or j is None:
            continue

        value = float(row.ppmi)

        rows.extend(
            [i, j]
        )
        cols.extend(
            [j, i]
        )
        values.extend(
            [value, value]
        )

    matrix = coo_matrix(
        (
            values,
            (rows, cols),
        ),
        shape=(
            len(card_ids),
            len(card_ids),
        ),
        dtype=np.float64,
    ).tocsr()

    return matrix


# -----------------------------------------------------------------------------
# SVD
# -----------------------------------------------------------------------------

def fit_decomposition(
    ppmi_matrix,
    requested_components: int,
    random_state: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    TruncatedSVD,
]:
    """
    Fit:

        M ~= U Sigma V^T

    sklearn TruncatedSVD.fit_transform() returns U Sigma.

    Recover U by dividing each output dimension by its corresponding
    singular value.
    """

    if (
        ppmi_matrix.shape[0] < 2
        or ppmi_matrix.nnz == 0
    ):
        raise ValueError(
            "Not enough positive PPMI structure "
            "to fit SVD"
        )

    components = min(
        requested_components,
        ppmi_matrix.shape[1] - 1,
    )

    if components < 1:
        raise ValueError(
            "SVD requires at least one component"
        )

    if components != requested_components:
        LOG.warning(
            "Reducing SVD components from %d to %d "
            "because only %d cards are eligible.",
            requested_components,
            components,
            ppmi_matrix.shape[1],
        )

    model = TruncatedSVD(
        n_components=components,
        random_state=random_state,
    )

    # sklearn gives us U @ Sigma.
    u_sigma = model.fit_transform(
        ppmi_matrix
    )

    sigma = np.asarray(
        model.singular_values_,
        dtype=np.float64,
    )

    # Recover U.
    u = np.zeros_like(
        u_sigma,
        dtype=np.float64,
    )

    nonzero = sigma > 0

    u[:, nonzero] = (
        u_sigma[:, nonzero]
        / sigma[nonzero]
    )

    if not np.isfinite(u).all():
        raise ValueError(
            "U contains non-finite values"
        )

    if not np.isfinite(sigma).all():
        raise ValueError(
            "Sigma contains non-finite values"
        )

    return (
        u,
        sigma,
        model,
    )


# -----------------------------------------------------------------------------
# Artifact persistence
# -----------------------------------------------------------------------------

def save_decomposition(
    *,
    artifact_path: Path,
    u: np.ndarray,
    sigma: np.ndarray,
    card_ids: np.ndarray,
    card_names: np.ndarray,
    metadata: dict,
) -> None:
    """
    Store all decomposition data in one compressed NumPy archive.

    The resulting .npz contains:

        U
        sigma
        card_ids
        card_names
        metadata
    """

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        artifact_path,
        U=u,
        sigma=sigma,
        card_ids=card_ids,
        card_names=card_names,
        metadata=np.asarray(
            json.dumps(metadata)
        ),
    )


# -----------------------------------------------------------------------------
# Main generation
# -----------------------------------------------------------------------------

def generate_decomposition(
    *,
    db_path: Path,
    artifact_path: Path,
    deck_size: int,
    cards_seen: int,
    min_pair_prob: float,
    svd_components: int,
    random_state: int,
    exclude_basics: bool,
) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            db_path
        )

    if deck_size <= 0:
        raise ValueError(
            "deck_size must be positive"
        )

    if not 0 < cards_seen <= deck_size:
        raise ValueError(
            "cards_seen must be between "
            "1 and deck_size"
        )

    if min_pair_prob < 0:
        raise ValueError(
            "min_pair_prob cannot be negative"
        )

    if svd_components < 1:
        raise ValueError(
            "svd_components must be positive"
        )

    basic_filter = build_basic_filter(
        exclude_basics
    )

    LOG.info(
        "Database: %s",
        db_path.resolve(),
    )

    LOG.info(
        "Artifact: %s",
        artifact_path.resolve(),
    )

    LOG.info(
        "Hyperparameters: deck_size=%d "
        "cards_seen=%d "
        "min_pair_prob=%g "
        "components=%d "
        "exclude_basics=%s",
        deck_size,
        cards_seen,
        min_pair_prob,
        svd_components,
        exclude_basics,
    )

    # ------------------------------------------------------------------
    # Load data and construct PPMI matrix
    # ------------------------------------------------------------------

    with duckdb.connect(
        str(db_path),
        read_only=True,
    ) as con:

        embedding_cards = (
            load_embedding_card_table(
                con,
                basic_filter,
            )
        )

        if embedding_cards.empty:
            raise ValueError(
                "No maindeck cards were found "
                "in deck_build_cards"
            )

        LOG.info(
            "Eligible decklist cards: %d",
            len(embedding_cards),
        )

        pairs = load_ppmi_pairs(
            con,
            deck_size=deck_size,
            cards_seen=cards_seen,
            min_pair_prob=min_pair_prob,
            basic_filter=basic_filter,
        )

    LOG.info(
        "Positive-PMI pairs: %d",
        len(pairs),
    )

    card_ids = np.asarray(
        [
            int(card_id)
            for card_id
            in embedding_cards[
                "card_id"
            ].tolist()
        ],
        dtype=np.int64,
    )

    card_names = np.asarray(
        embedding_cards[
            "card_name"
        ].astype(str).tolist(),
        dtype=np.str_,
    )

    ppmi_matrix = build_ppmi_matrix(
        pairs,
        card_ids.tolist(),
    )

    LOG.info(
        "PPMI matrix: %d x %d "
        "with %d non-zero entries.",
        ppmi_matrix.shape[0],
        ppmi_matrix.shape[1],
        ppmi_matrix.nnz,
    )

    # ------------------------------------------------------------------
    # Decompose
    # ------------------------------------------------------------------

    u, sigma, model = fit_decomposition(
        ppmi_matrix,
        svd_components,
        random_state,
    )

    explained_variance = float(
        model.explained_variance_ratio_.sum()
    )

    LOG.info(
        "SVD explained variance: %.4f",
        explained_variance,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata = {
        "decomposition": "truncated_svd",
        "source_matrix": "deck_ppmi",
        "matrix_rows": int(
            ppmi_matrix.shape[0]
        ),
        "matrix_columns": int(
            ppmi_matrix.shape[1]
        ),
        "matrix_nonzero": int(
            ppmi_matrix.nnz
        ),
        "components": int(
            u.shape[1]
        ),
        "deck_size": deck_size,
        "cards_seen": cards_seen,
        "min_pair_prob": min_pair_prob,
        "exclude_basics": exclude_basics,
        "random_state": random_state,
        "explained_variance": (
            explained_variance
        ),
    }

    # ------------------------------------------------------------------
    # Save one artifact
    # ------------------------------------------------------------------

    save_decomposition(
        artifact_path=artifact_path,
        u=u,
        sigma=sigma,
        card_ids=card_ids,
        card_names=card_names,
        metadata=metadata,
    )

    LOG.info(
        "Saved decomposition to %s",
        artifact_path,
    )

    LOG.info(
        "U shape: %s",
        u.shape,
    )

    LOG.info(
        "Sigma shape: %s",
        sigma.shape,
    )

    LOG.info(
        "Cards: %d",
        len(card_ids),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deck-context PPMI matrix, "
            "fit a truncated SVD decomposition, "
            "and store U and Sigma in a NumPy archive."
        )
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
    )

    parser.add_argument(
        "--artifact",
        type=Path,
        default=(
            ARTIFACT_DIR
            / ARTIFACT_FILE
        ),
    )

    parser.add_argument(
        "--deck-size",
        type=int,
        default=DECK_SIZE,
    )

    parser.add_argument(
        "--cards-seen",
        type=int,
        default=CARDS_SEEN,
    )

    parser.add_argument(
        "--min-pair-prob",
        type=float,
        default=MIN_PAIR_PROB,
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=SVD_COMPONENTS,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
    )

    parser.add_argument(
        "--include-basics",
        action="store_true",
        help=(
            "Include basic lands in the PPMI matrix. "
            "The default excludes them."
        ),
    )

    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level.upper(),
        ),
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        ),
    )

    generate_decomposition(
        db_path=args.db,
        artifact_path=args.artifact,
        deck_size=args.deck_size,
        cards_seen=args.cards_seen,
        min_pair_prob=args.min_pair_prob,
        svd_components=args.dimensions,
        random_state=args.random_state,
        exclude_basics=(
            not args.include_basics
        ),
    )


if __name__ == "__main__":
    main()