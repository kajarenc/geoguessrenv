# Gymnasium Env + GeoGuessr-like Environment

This repo contains custom Gymnasium environments and wrappers, including a GeoGuessr-like panorama navigation environment and an OpenAI-powered base agent.

## Quickstart (using uv)

```bash
# 1) Clone the repo
git clone https://github.com/kajarenc/geoguessrenv.git
cd geoguessrenv

# 2) Create & activate a virtual environment for this project
uv venv
source .venv/bin/activate

# 3) Install the project and dependencies
uv pip install -e .

# 4) Configure environment variables
cp .env.example .env
# Edit .env and set your keys (OPENAI_API_KEY required for the agent)

# 5a) Run the manual demo
uv run python geoguessr_env_demo.py

# 5b) Run the OpenAI agent (requires OPENAI_API_KEY)
uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 10 --input_lat 47.620908 --input_lon -122.353508 --render
```


### Note
For --input_lat and --input_lon use 6 digits after the decimal point format.

## Demos

- `geoguessr_env_demo.py`: Basic example to test GeoGuessrWorldEnv environment.

## OpenAI Agent Runner

Run an OpenAI-powered base agent that uses a vision model to navigate and answer with lat/lon.

Requirements:
- Set `OPENAI_API_KEY` in your environment.

You can also place it in a `.env` file in the project root:

```bash
cp .env.example .env
echo 'OPENAI_API_KEY=sk-...' >> .env
```

Example:

```bash
export OPENAI_API_KEY=sk-...
uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 10 --input_lat 47.620908 --input_lon -122.353508 --render
```

Arguments:
- `--model`: OpenAI vision model (default `gpt-4o`).
- `--max_nav_steps`: Maximum navigation steps before answering (default `10`).
- `--image_width/--image_height`: Image size sent to the model (defaults `1024x512`).
- `--input_lat/--input_lon`: Seed location for fetching nearby panoramas.
- `--cache_root`: Where the environment stores images/metadata (defaults to `cache`).
- `--cache_dir`: Optional local cache directory for agent responses.
- `--render`: Show a Pygame window while running.


Run the baseline agent with geofence:
```
uv run python -m geoguess_env.run_baseline \
  --mode online \
  --provider gsv \
  --episodes 2 \
  --geofence geofences/seattle_5km.json \
  --cache ./cache \
  --seed 22 \
  --out results_online.csv
```

