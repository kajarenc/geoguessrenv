uv run python -m geoguess_env.run_baseline \
  --mode online \
  --provider gsv \
  --episodes 2 \
  --geofence geofences/seattle_5km.json \
  --cache ./cache \
  --seed 42 \
  --freeze-run ./cache/replays/session_123.json \
  --out results_online.csv


uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 10 --input_lat 52.358836 --input_lon 4.880845 --render