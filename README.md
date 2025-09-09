# Gymnasium Env + GeoGuessr-like Environment

This repo contains custom Gymnasium environments and wrappers, including a GeoGuessr-like panorama navigation environment and an OpenAI-powered base agent.

## Installation

```bash
uv pip install -e .
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
uv run python scripts/run_openai_agent.py --render --model gpt-4o --max_nav_steps 40 --image_width 1024 --image_height 512
```

Arguments:
- `--model`: OpenAI vision model (default `gpt-4o`).
- `--max_nav_steps`: Maximum navigation steps before answering (default `40`).
- `--image_width/--image_height`: Image size sent to the model (defaults `1024x512`).
- `--input_lat/--input_lon`: Seed location for fetching nearby panoramas.
- `--cache_root`: Where the environment stores images/metadata (defaults to `tempcache`).
- `--cache_dir`: Optional local cache directory for agent responses.
- `--render`: Show a Pygame window while running.

