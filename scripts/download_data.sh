#!/usr/bin/env bash

set -euo pipefail

SET="${1:-Cube_-_Powered}"
FORMAT="${2:-PremierDraft}"

if [[ -z "$SET" ]]; then
    echo "Usage: $0 <SET> [FORMAT]"
    echo "Example: $0 EOE PremierDraft"
    exit 1
fi

DATA_DIR="./data"
BASE_URL="https://17lands-public.s3.amazonaws.com/analysis_data"

mkdir -p "$DATA_DIR"

GAME_FILE="game_data_public.${SET}.${FORMAT}.csv.gz"
REPLAY_FILE="replay_data_public.${SET}.${FORMAT}.csv.gz"

CARDS_FILE="cards.csv"
ABILITIES_FILE="abilities.csv"

echo "Downloading 17Lands data for ${SET} (${FORMAT})..."

# ------------------------------------------------------------------
# Game data
# ------------------------------------------------------------------

curl -fL \
    "${BASE_URL}/game_data/${GAME_FILE}" \
    -o "${DATA_DIR}/${GAME_FILE}"

# ------------------------------------------------------------------
# Replay data
# ------------------------------------------------------------------

curl -fL \
    "${BASE_URL}/replay_data/${REPLAY_FILE}" \
    -o "${DATA_DIR}/${REPLAY_FILE}"

# ------------------------------------------------------------------
# Card and ability lookup data
# ------------------------------------------------------------------

curl -fL \
    "${BASE_URL}/cards/${CARDS_FILE}" \
    -o "${DATA_DIR}/${CARDS_FILE}"

curl -fL \
    "${BASE_URL}/abilities/${ABILITIES_FILE}" \
    -o "${DATA_DIR}/${ABILITIES_FILE}"

echo
echo "Downloads complete:"
echo "  ${DATA_DIR}/${GAME_FILE}"
echo "  ${DATA_DIR}/${REPLAY_FILE}"
echo "  ${DATA_DIR}/${CARDS_FILE}"
echo "  ${DATA_DIR}/${ABILITIES_FILE}"