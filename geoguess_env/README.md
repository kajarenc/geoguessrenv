# GeoGuess Environment - Baseline Implementation

This package implements the baseline components specified in TaskDescription.md for the GeoGuessr-style RL environment.

## Components Implemented

### ✅ VLMBroker (`vlm_broker.py`)
- Standardized I/O for VLM interactions
- Builds prompts explaining the task and allowed actions
- Parses VLM responses into valid actions
- Handles malformed JSON with safe fallbacks
- Action format: `{"op":"click","value":[640,256]}` or `{"op":"answer","value":[47.6205,-122.3493]}`

### ✅ BaselineAgent (`baseline_agent.py`)
- Simple arrow-following agent as specified
- Sweeps screen positions to find navigation arrows
- Follows arrows for K steps or until loop detection (pano_id repeats)
- Answers with simple heuristic using continent centroids
- Configurable max navigation steps and sweep patterns

### ✅ CLI Runner (`run_baseline.py`)
- Command-line interface for sampling episodes while caching assets
- CSV output with episode results
- Respects existing cache entries to avoid redundant network calls

## Usage Examples

### Basic Demo
```bash
uv run python geoguessr_env_demo.py
```

### CLI Usage
```bash
uv run python -m geoguess_env.run_baseline \
  --provider gsv \
  --episodes 30 \
  --cache ./cache \
  --seed 123 \
  --out results_online.csv
```

## Integration with Existing Environment

The baseline implementation works seamlessly with the existing `GeoGuessrWorldEnv`:

- ✅ Uses correct action format with `"value"` key
- ✅ Handles click navigation and answer termination
- ✅ Receives appropriate rewards (0.0 for clicks, distance-based for answers)
- ✅ Works with existing caching and image loading system
- ✅ Compatible with existing test suite

## Action Format

All components use the standardized action format:

**Click (navigation):**
```json
{"op":"click","value":[640,256]}
```

**Answer (termination):**
```json
{"op":"answer","value":[47.6205,-122.3493]}
```

## CSV Output Format

Results are saved with the following columns:
```
episode,pano_id,gt_lat,gt_lon,guess_lat,guess_lon,distance_km,score,steps
```

## Next Steps

This baseline implementation provides the foundation for:

1. **Enhanced geofencing** - Add proper geographic sampling boundaries
2. **Deterministic replay** - Full session replay with exact episode reproduction
3. **Cache structure alignment** - Move to `cache/{images,metadata}/` structure
4. **Package restructuring** - Create proper `geoguess_env/` package structure
5. **Advanced agents** - Build on the baseline for more sophisticated navigation strategies

The current implementation demonstrates all core concepts working together and provides a solid foundation for further development toward full TaskDescription.md compliance.
