Project-specific development guidelines

Audience: Advanced contributors to this repository. The points below capture how this project is actually built, configured, and tested in practice, with caveats that matter for day-to-day development.

1) Build and configuration

- Python/runtime
  - Requires Python >= 3.10 (see pyproject.toml). Tested locally with Python 3.12 as well.
  - Build backend: hatchling; primary dev workflow uses uv for env + installs.

- Recommended setup (uv)
  - Create venv and install in editable mode:
    - uv venv
    - source .venv/bin/activate
    - uv pip install -e .

- Environment variables and external services
  - OPENAI_API_KEY is required only when running the OpenAI agent (scripts/run_openai_agent.py). The test suite does not require it.
  - The environment supports Google Street View data via streetview/streetlevel; normal test runs do not fetch network data. Interactive demos may download and cache assets under ./cache.
  - A .env can be used at repo root (see README). For agents: cp .env.example .env and set OPENAI_API_KEY.

- Rendering/headless notes (pygame)
  - The environment can render via pygame for demos. Tests do not require a display and pass headlessly. If you add new tests that initialize pygame display surfaces, prefer headless mode. If needed, export SDL_VIDEODRIVER=dummy prior to running tests in CI.

- Caching and data
  - Image and metadata cache defaults to ./cache. Tests do not rely on networked cache; they either stub providers or use synthetic data.

2) Testing

- Test runner
  - Pytest is included as a project dependency (pyproject.toml). Typical invocations:
    - uv run pytest -q
  - Current suite status (validated locally before writing this doc): 134 passed, 1 warning. A pygame pkg_resources deprecation warning is benign.

- Scope and isolation
  - Ensure tests remain offline: mock or stub any network calls to providers or the OpenAI API. The existing tests demonstrate this approach (e.g., agents.openai_agent is subclassed with a stubbed _chat_completions).
  - Avoid writing to ./cache in unit tests unless the test owns and cleans the directory. Prefer temporary directories via pytest tmp_path.

- Linting/formatting and typing checks (optional but recommended locally)
  - Ruff is configured in pyproject.toml (line-length 88, target-version py311, import sort). Run if installed:
    - ruff check .
    - ruff format .
  - Optional static checks (ty): uv run ty check geoguess_env/ tests/

- Adding a new test
  - Place tests under tests/ with names like test_*.py. Keep them deterministic and offline. If interacting with the environment, stub providers and asset I/O.
  - Example (verified locally):
    - File: tests/test_smoke_example.py
      from geoguess_env.geometry_utils import GeometryUtils

      def test_haversine_zero_distance():
          assert GeometryUtils.haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0

      def test_haversine_one_degree_longitude_at_equator():
          # Roughly 111.195 km for 1 degree of longitude at the equator
          d = GeometryUtils.haversine_distance(0.0, 0.0, 0.0, 1.0)
          assert 111.0 < d < 111.5
    - Run:
      - pytest -q
    - Expected: tests pass with the rest of the suite. After validation, remove the example file if it is not meant to live in the repo.

- Running a subset
  - Single file: pytest tests/test_geoguessr_env.py -q
  - Single test: pytest tests/test_geoguessr_env.py::test_click_action -q

3) Additional development notes

- Project layout
  - geoguess_env/: core environment, action parsing, providers, geometry utils, asset manager, config types.
  - agents/: base and OpenAI-driven agents, plus CLI runners in scripts/.
  - tests/: comprehensive unit tests. They prioritize isolation and non-network determinism.

- Code style and conventions
  - Follow ruff rules configured in pyproject.toml (quote-style double, line length 88). Type hints are used throughout; prefer precise typing and Pydantic models where applicable.
  - Keep functions small and explicit; avoid hidden global state beyond controlled caches.

- Provider and asset caveats
  - GoogleStreetViewProvider and related classes should never be hit by unit tests with live network calls. Use stubs or inject providers via configuration to keep tests hermetic.
  - AssetManager writes and reads from the cache; in tests, redirect to tmp dirs. Verify asset existence via helper methods instead of touching filesystem directly.

- Environment behavior highlights
  - step(action): supports two ops—click navigation and answer submission—parsed via ActionParser. Answer submissions compute reward with GeometryUtils; info includes guess coords, distance_km, and score.
  - max_steps (GeoGuessrConfig) triggers truncated episodes when exceeded without answer.
  - For new logic, keep reward calculations and termination flags consistent with tests; prefer adding targeted tests in tests/test_geoguessr_env.py or a new file.

- Demos and runners
  - Manual demo: uv run python geoguessr_env_demo.py (renders via pygame).
  - OpenAI agent demo (requires OPENAI_API_KEY):
    uv run python scripts/run_openai_agent.py \
      --model gpt-4o \
      --max_nav_steps 10 \
      --input_lat 47.620908 --input_lon -122.353508 \
      --render
  - Baseline agent with geofence:
    uv run python -m geoguess_env.run_baseline \
      --provider gsv \
      --episodes 2 \
      --geofence geofences/seattle_15km.json \
      --cache ./cache \
      --seed 123 \
      --out results_online.csv

- CI/repro tips
  - Pin via uv.lock when using uv for consistent environments. The build system is hatchling; packaging uses the geoguess_env package in pyproject.
  - If a headless CI needs to import pygame without a display, set SDL_VIDEODRIVER=dummy and avoid surfacing windows in tests.

This document was validated by actually installing and running the full test suite locally, plus a temporary smoke test that passed; only this file (.junie/guidelines.md) was kept after cleanup.