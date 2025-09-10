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
 uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 20 --input_lat 47.618566 --input_lon -122.354386 --render
```


## Demos

- `geoguessr_env_demo.py`: Manually interact with the GeoGuessr-like environment.

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
uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 20 --input_lat 47.618566 --input_lon -122.354386 --render
```

Arguments:
- `--model`: OpenAI vision model (default `gpt-4o`).
- `--max_nav_steps`: Maximum navigation steps before answering (default `40`).
- `--image_width/--image_height`: Image size sent to the model (defaults `1024x512`).
- `--input_lat/--input_lon`: Seed location for fetching nearby panoramas.
- `--cache_root`: Where the environment stores images/metadata (defaults to `tempcache`).
- `--cache_dir`: Optional local cache directory for agent responses.
- `--render`: Show a Pygame window while running.

