# Agent Integration Guide

`GeoGuessrEnv` (`geoguess_env/geoguessr_env.py`) is the primary artifact in this repository. Every shipped agent exists to exercise that environment, verify its APIs, and offer reproducible entry points for research or demos. This guide captures the environment touchpoints, the project tooling expected for agent work, the reference agents already implemented, and practical advice for extending the system.

## Environment & Tooling Baseline
- **Runtime**: Python 3.11+ (see `pyproject.toml`).
- **Dependency manager**: [`uv`](https://github.com/astral-sh/uv) only. Install the project with `uv pip install -e .` inside the repository root.
- **Formatting & linting**: Ruff (`uv run ruff check . --fix`, `uv run ruff format .`).
- **Testing**: pytest (`uv run pytest -v`). Add targeted tests under `tests/` for any new agent behaviors.
- **Environment variables**: Use `.env` and `dotenv` when credentials are required (`OPENAI_API_KEY`, `GOOGLE_MAPS_API_KEY`).
- **Cache layout**: Persist agent artifacts beneath the configured cache root (defaults to `./cache/`). Follow the image/metadata/replay structure documented in `TaskDescription.md` where possible.

## GeoGuessrEnv Interaction Model
Agents interact with the environment through the standard Gymnasium loop:
- **Observation**: `reset()` and `step()` return `{"image": np.uint8[H, W, 3]}` arrays. Dimensions follow the environment's `render_config` (default 1024×512).
- **Info payload**: Every step includes navigation metadata:
  - `links`: Clickable targets with `screen_xy`, `heading_deg`, `_rel_heading_deg`, and `conf` ready for hit-testing.
  - `pose`: Camera orientation (`yaw_deg` / `heading_deg`).
  - `steps`, `pano_id`, `gt_lat`, `gt_lon`, `distance_km`, and `score` expose progression and scoring state.
- **Actions**: You must return JSON-compatible dicts using the schema enforced by `geoguess_env.action_parser`:
  - Click: `{"op": "click", "click": [x, y]}` (navigation only, zero reward).
  - Answer: `{"op": "answer", "answer": [lat_deg, lon_deg]}` (terminates the episode and scores).
- **Step budgeting**: Respect both the environment's `max_steps` and your agent's configuration (e.g., `AgentConfig.max_nav_steps`). Always provide a fallback answer once the budget is exhausted.

## Project Code Map for Agents
- `agents/base.py`: Dataclasses (`AgentConfig`, `AgentAction`) plus the `BaseAgent` interface used by VLM-driven implementations.
- `agents/openai_agent.py`: OpenAIVisionAgent showcasing multimodal prompting through the environment's `VLMBroker`.
- `agents/openai_models.py`: Pydantic schemas that validate OpenAI tool calls (`click`, `answer`).
- `agents/utils.py`: Image encoding, fingerprinting, and cache helpers for model responses.
- `geoguess_env/baseline_agent.py`: Offline baseline that sweeps click positions and answers with continent-level heuristics.
- `scripts/run_openai_agent.py`: CLI driver for the OpenAI demo agent.
- `geoguess_env/run_baseline.py`: CLI runner for the deterministic baseline agent.

## Reference Agents
### BaselineAgent (offline-friendly)
- **File**: `geoguess_env/baseline_agent.py`
- **Usage**:
  ```bash
  uv run python -m geoguess_env.run_baseline \
    --episodes 10 \
    --cache ./cache \
    --out results.csv
  ```
- **Highlights**: Horizontal click sweep, loop detection via visited pano IDs, heuristic answers anchored on continent centroids.

### OpenAIVisionAgent (multimodal)
- **File**: `agents/openai_agent.py`
- **Usage**:
  ```bash
  uv run python scripts/run_openai_agent.py \
    --model gpt-4o \
    --max_nav_steps 10 \
    --input_lat 47.620908 \
    --input_lon -122.353508 \
    --render
  ```
- **Highlights**: Structured prompts via `geoguess_env.vlm_broker.VLMBroker`, optional caching (`--cache_dir`), Pydantic validation of tool outputs, history-aware prompting.
- **Prerequisites**: `OPENAI_API_KEY` (export or store in `.env`), `uv pip install -e .` to pull the optional OpenAI dependency.

## Building New Agents
- Start from `BaseAgent` or wrap it; extend `AgentConfig` for new hyperparameters rather than adding ad-hoc kwargs.
- Consume `info["links"]` to pick navigation targets; clamp click coordinates to the observation bounds before returning them.
- Keep implementations PEP 8 compliant and Ruff-clean—descriptive naming should make the code self-documenting.
- Persist any learned state or cache files inside `cache/agent_<name>/` (or another user-provided directory) and write JSON with atomic replacements (`agents.utils.cache_put`).
- For external services, load credentials lazily and emit actionable error messages if prerequisites are missing.

## Quality & Testing Expectations
- **Unit tests**: Place under `tests/` and run with `uv run pytest -v`. Mock network calls to maintain deterministic behavior.
- **Static checks**: `uv run ruff check .` and `uv run ruff format .` should pass before submitting changes.
- **CI alignment**: The GitHub Actions workflow mirrors these commands across Python 3.10–3.12; match that matrix locally when feasible.
- **Data logging**: When writing CSVs or JSON, format lat/lon with six decimal places (`f"{lat:.6f}"`) to align with reporting scripts.

## Troubleshooting
- **Empty `links` list**: The panorama has no outgoing edges—agents should answer immediately to avoid infinite loops.
- **Click misalignment**: Verify your `render_config` matches the dimensions assumed by the agent; mismatched widths lead to off-target clicks.
- **Cache misses**: Ensure `--cache` (baseline) or `--cache_dir` (OpenAI agent) points to a writable directory. Remove stale cache files if prompt fingerprints change.
- **Credential errors**: Confirm environment variables (`OPENAI_API_KEY`, `GOOGLE_MAPS_API_KEY`) are set and that `.env` is loaded via `dotenv` where applicable.

`GeoGuessrEnv` remains the source of truth. Treat agents as thin adapters layered on top of it, keep observer access read-only, and prefer deterministic strategies that make regression testing straightforward. For deeper architectural details, consult `TaskDescription.md`, `CLAUDE.md`, and the environment implementation in `geoguess_env/geoguessr_env.py`.
