uv run python -m geoguess_env.run_baseline \
  --mode online \
  --provider gsv \
  --episodes 2 \
  --geofence geofences/seattle_10km.json \
  --cache ./cache \
  --seed 123 \
  --freeze-run ./cache/replays/session_123.json \
  --out results_online.csv