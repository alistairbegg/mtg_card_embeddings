from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DB_PATH = Path("data/17lands.duckdb")

# Download in chunks so the database is never loaded fully into memory.
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

# Used when verifying the downloaded file.
HASH_CHUNK_SIZE = 8 * 1024 * 1024


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


# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------

def download_database(
    url: str,
    expected_sha256: str | None = None,
) -> None:
    """
    Download the database from a presigned S3 URL.

    The file is downloaded to a temporary location first. The existing
    local database is replaced only after the download has completed
    successfully and any requested checksum verification has passed.
    """
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = DB_PATH.with_suffix(
        ".duckdb.download"
    )

    # Remove an incomplete download from a previous failed attempt.
    temp_path.unlink(
        missing_ok=True,
    )

    print(
        f"Downloading database -> {temp_path}"
    )

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, None),
        ) as response:
            response.raise_for_status()

            total_bytes = int(
                response.headers.get(
                    "content-length",
                    0,
                )
            )

            downloaded = 0

            with temp_path.open("wb") as f:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE,
                ):
                    if not chunk:
                        continue

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_bytes:
                        percent = (
                            downloaded
                            / total_bytes
                            * 100
                        )

                        print(
                            f"\rDownloaded "
                            f"{downloaded / 1024**3:.2f} GB "
                            f"({percent:.1f}%)",
                            end="",
                            flush=True,
                        )

            if total_bytes:
                print()

    except Exception:
        temp_path.unlink(
            missing_ok=True,
        )
        raise

    # --------------------------------------------------------------
    # Verify checksum
    # --------------------------------------------------------------

    if expected_sha256 is not None:
        print(
            "Verifying SHA-256 checksum..."
        )

        actual_sha256 = sha256_file(
            temp_path
        )

        if actual_sha256.lower() != expected_sha256.lower():
            temp_path.unlink(
                missing_ok=True,
            )

            raise RuntimeError(
                "Downloaded database checksum "
                "does not match expected checksum.\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_sha256}"
            )

        print(
            "Checksum verified."
        )

    # --------------------------------------------------------------
    # Replace local database
    # --------------------------------------------------------------

    # Path.replace() is used only after the full download has
    # succeeded, so a failed download cannot destroy the existing DB.
    temp_path.replace(
        DB_PATH
    )

    print()
    print(
        f"Database saved to {DB_PATH}"
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the shared DuckDB database "
            "from a presigned S3 URL."
        )
    )

    parser.add_argument(
        "url",
        help="Presigned S3 download URL.",
    )

    parser.add_argument(
        "--sha256",
        default=None,
        help=(
            "Optional expected SHA-256 checksum "
            "for the database."
        ),
    )

    args = parser.parse_args()

    download_database(
        url=args.url,
        expected_sha256=args.sha256,
    )


if __name__ == "__main__":
    main()