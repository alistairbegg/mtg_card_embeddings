from __future__ import annotations

import duckdb
import pandas as pd

DB_URL = (
    "https://alistairbegg-personal-projects.s3.eu-west-2.amazonaws.com/mtg_card_embeddings/data/17lands.duckdb"
)


def get_db_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")

    con.execute(
        f"""
        ATTACH '{DB_URL}'
        AS mtg (READ_ONLY)
        """
    )

    return con


def query_db(
    con: duckdb.DuckDBPyConnection,
    query: str,
    params: list | tuple | None = None,
) -> pd.DataFrame:
    if params is None:
        return con.execute(query).fetchdf()

    return con.execute(query, params).fetchdf()