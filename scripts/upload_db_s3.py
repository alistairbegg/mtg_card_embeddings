from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import boto3
import duckdb
from botocore.config import Config


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DB_PATH = Path("data/17lands.duckdb")

S3_BUCKET = "alistairbegg-personal-projects"
S3_PREFIX = "mtg_card_embeddings/data"
AWS_REGION = "eu-west-2"

DATABASE_KEY = f"{S3_PREFIX}/17lands.duckdb"
MANIFEST_KEY = f"{S3_PREFIX}/manifest.json"

# Presigned URL lifetime.
#
# Seven days is convenient for a small project where you can simply
# generate a new URL whenever you publish a new database snapshot.
PRESIGNED_URL_EXPIRY_SECONDS = 7 * 24 * 60 * 60

# Read the database in chunks when calculating its checksum rather
# than loading the whole file into memory.
HASH_CHUNK_SIZE = 8 * 1024 * 1024


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the local DuckDB snapshot to S3 and generate a "
            "presigned download URL."
        )
    )

    parser.add_argument(
        "--url-only",
        action="store_true",
        help=(
            "Do not upload anything. Generate a new presigned URL "
            "for the database already stored in S3."
        ),
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# AWS
# ------------------------------------------------------------------

def create_s3_client():
    """
    Create an S3 client using AWS Signature Version 4.

    Credentials are resolved normally by boto3. For example:

        AWS_PROFILE=personal uv run python -m scripts.upload_db_s3

    On EC2, an attached IAM role can be used instead.
    """
    session = boto3.Session()

    return session.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version="s3v4"),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit() -> str | None:
    """Return the current Git commit when running inside a repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


# ------------------------------------------------------------------
# Prepare database
# ------------------------------------------------------------------

def checkpoint_database() -> None:
    """
    Flush DuckDB's WAL into the main database file.

    The uploaded .duckdb file should therefore be a self-contained,
    consistent database snapshot.
    """
    print(f"Checkpointing {DB_PATH}...")

    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("CHECKPOINT")


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

def build_manifest() -> dict:
    """Build metadata describing the uploaded database snapshot."""
    print("Calculating database checksum...")

    return {
        "database": DB_PATH.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": DB_PATH.stat().st_size,
        "sha256": sha256_file(DB_PATH),
        "git_commit": get_git_commit(),
    }


# ------------------------------------------------------------------
# S3 upload
# ------------------------------------------------------------------

def upload_database(
    s3,
    manifest: dict,
) -> None:
    """Upload the DuckDB database and its manifest."""
    print(
        f"Uploading {DB_PATH} -> "
        f"s3://{S3_BUCKET}/{DATABASE_KEY}"
    )

    # boto3 automatically uses multipart uploads for sufficiently
    # large files, so this is suitable for a large DuckDB database.
    s3.upload_file(
        str(DB_PATH),
        S3_BUCKET,
        DATABASE_KEY,
    )

    print(
        f"Uploading manifest -> "
        f"s3://{S3_BUCKET}/{MANIFEST_KEY}"
    )

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=MANIFEST_KEY,
        Body=json.dumps(
            manifest,
            indent=2,
        ).encode("utf-8"),
        ContentType="application/json",
    )


# ------------------------------------------------------------------
# Presigned URL
# ------------------------------------------------------------------

def generate_download_url(s3) -> str:
    """
    Generate a temporary download URL.

    The person using this URL does not need their own AWS credentials.
    """
    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": S3_BUCKET,
            "Key": DATABASE_KEY,
        },
        ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    s3 = create_s3_client()

    # --------------------------------------------------------------
    # URL-only mode
    # --------------------------------------------------------------

    if args.url_only:
        download_url = generate_download_url(s3)

        print("Presigned download URL:")
        print(download_url)

        print()
        print(
            "This URL expires in "
            f"{PRESIGNED_URL_EXPIRY_SECONDS // 86400} days."
        )

        return

    # --------------------------------------------------------------
    # Normal upload mode
    # --------------------------------------------------------------

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database does not exist: {DB_PATH}"
        )

    checkpoint_database()

    manifest = build_manifest()

    upload_database(
        s3,
        manifest,
    )

    download_url = generate_download_url(s3)

    print()
    print("Upload complete.")
    print()

    print("Snapshot:")
    print(json.dumps(manifest, indent=2))

    print()
    print("Presigned download URL:")
    print(download_url)

    print()
    print(
        "This URL expires in "
        f"{PRESIGNED_URL_EXPIRY_SECONDS // 86400} days."
    )


if __name__ == "__main__":
    main()