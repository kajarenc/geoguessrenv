# GeoGuessrEnv Copilot Instructions
## Project snapshot
- Core environment: `geoguess_env/geoguessr_env.py` couples `GeoGuessrConfig`, `AssetManager`, and `ActionParser` to drive Gymnasium episodes returning `{"image": np.uint8[H, W, 3]}` observations.
- `AssetManager.prepare_graph(...)` hydrates cached panoramas, retries downloads, and blocklists failures in `cache/metadata/failed_panoramas.json`; repeated calls round lat/lon to six decimals when resolving panoramas.
- Configuration flows through `GeoGuessrConfig.from_dict`, which normalizes `cache_root`, validates coordinates, and nests provider/render/navigation configs—keep new settings inside those dataclasses.
- Navigation budget is two-tiered: env `max_steps` (default 40) and agent `AgentConfig.max_nav_steps`; always furnish a fallback answer when either limit is hit.
## Agent patterns
- The only accepted action schema is `{"op": "click"|"answer", "value": [...]}`; `ActionParser` clamps coordinates and falls back to center-click, so match that shape when producing actions or tests.
- Base agent contracts live in `agents/base.py`; extend `AgentConfig` instead of sprinkling kwargs, and honour cached response helpers in `agents/utils.py` (`cache_get`, `cache_put`, `compute_prompt_fingerprint`).
- `agents/openai_agent.py` funnels OpenAI tool calls through `VLMBroker`; snap clicks to provided `info["links"]` (see `_snap_to_nearest_link`) and preserve the history window for context-aware prompts.
- Offline baseline behaviour is in `geoguess_env/baseline_agent.py`; reuse its link sweep and continent heuristics when you need deterministic fallbacks.
- Persist agent artifacts under `cache/agent_<name>/` and reuse JPEG/base64 helpers from `agents/utils.py` rather than re-encoding manually.
## Workflow essentials
- Manage dependencies with uv only: `uv pip install -e .`, then format/lint/typecheck via `uv run ruff format .`, `uv run ruff check . --fix`, and `uv run ty check .`.
- Quick demos: `uv run python geoguessr_env_demo.py` for manual play, or `uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 10 --render` (supply 6-decimal `--input_lat/--input_lon`).
- Baseline evaluation entry point: `uv run python -m geoguess_env.run_baseline --geofence geofences/seattle_15km.json --episodes 2 --cache ./cache --out results.csv`.
## Data & caching
- Cache layout is fixed: `cache/images/` for JPEGs, `cache/metadata/` for JSONL graphs, `cache/replays/` for episode logs; preserve it when adding new assets.
- `AssetManager` prunes invalid links and skips blocklisted panoramas—tests in `tests/test_env_retry_logic.py` depend on that retry + blocklist handshake.
- When fetching new data, call `AssetManager.preload_assets` or `prepare_graph`; direct file writes risk bypassing hash validation.
## Testing cues
- Unit tests mock asset loading heavily (see `tests/test_env_retry_logic.py` and `tests/test_integration.py`); prefer patching `prepare_graph` or `resolve_nearest_panorama` instead of touching disk.
- `tests/test_openai_agent_tools.py` asserts click snapping and value clamping; preserve `{"op": ..., "value": [...]}` outputs when refactoring the agent.
- Geometry utilities (`geoguess_env/geometry_utils.py`) back both reward and distance reporting—update corresponding tests if you touch haversine math or scoring.
