Refactor @geoguess_env/geoguessr_env.py according to best practices for building Gym compatible reinforcement learning environments.

Requirements:
- Improve the asset loading logic to make it more robust and modular.
- Document functions and classes with docstrings
- Add tests for @geoguess_env/geoguessr_env.py and data loading process.
- Also consultate with @TaskDescription.md and @Report.md. 

When I run a command with coordinates that not prefetched previoulsy, e.g. : 
uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 10  --input_lat 52.499494 --input_lon 13.378001 --render
The GeoGuessr environment doesn't fetch the actual metadata and images that required, and instead shows a black screen.
Please fix this issue, if metadata and images are not existing in the cache, get nearest pano, start bfs from there and 
fetch all required assets for environment creation, as it used to work before recent uncommited changes.
Think a lot1