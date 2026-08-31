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
## 4. Download Database
- `uv run python -m scripts.download_db [url]` *(I will generate temporary url)*
## 5. Run Python
- Use `uv run python ...` for all scripts
- Select .venv/bin/python as the notebook kernel/interpreter