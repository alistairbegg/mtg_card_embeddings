# Setup
## 1. Clone this Repo
- `git clone ...`
- `cd mtg_card_embeddings`
## 2. Install uv
### MacOS / Linux
- `curl -LsSf https://astral.sh/uv/install.sh | sh`
- or `brew install uv` *(macOS only)*
### Windows
- `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
## 3. Sync Env
- `uv sync`
## 4. Run Python
- Use `uv run python ...` for all scripts
- Select .venv/bin/python as the notebook kernel/interpreter

# Querying DB

- `from utils.db import get_db_connection, query_db`
- Open connection using `con = get_db_connection()`
- Query db using `query_db(con, query, params)`
- Close connection when done using `con.close()`

# Database Structure

```text
drafts
│
├── draft_picks
│     └── draft_pick_cards
│
├── deck_builds
│     └── deck_build_cards
│
└── games
      ├── game_players
      ├── game_card_stats
      ├── game_player_totals
      ├── candidate_hands
      │     └── candidate_hand_cards
      │
      └── turns
            ├── turn_player_state
            ├── turn_zone_cards
            └── events
```
### Main Foreign Key Links
```text
drafts
  └─ draft_id ──────────────> draft_picks
  └─ draft_id ──────────────> deck_builds
  └─ draft_id ──────────────> games

draft_picks
  └─ pick_id ───────────────> draft_pick_cards

deck_builds
  └─ build_id ──────────────> deck_build_cards
  └─ build_id ──────────────> games

games
  └─ game_id ───────────────> game_players
  └─ game_id ───────────────> game_card_stats
  └─ game_id ───────────────> game_player_totals
  └─ game_id ───────────────> candidate_hands
  └─ game_id ───────────────> turns

candidate_hands
  └─ hand_id ───────────────> candidate_hand_cards

turns
  └─ turn_id ───────────────> turn_player_state
  └─ turn_id ───────────────> turn_zone_cards
  └─ turn_id ───────────────> events
```
cards.card_id is the shared card key used across everything