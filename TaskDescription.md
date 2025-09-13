# **Take-Home: RL GeoGuessr-style Environment**

*(Online Sampling, Click Arrows to Navigate \+ Answer to Finish)*

**Goal**  
 Build a Gymnasium-compatible RL environment that samples a real street-level panorama online (e.g., Google Street View, Mapillary, KartaView), lets an agent **navigate only by clicking on the in-image navigation arrows** (forward/back along the provider’s link graph), and ends when the agent issues an **answer** giving `(lat, lon)`. **Clicks never yield reward**; only the final answer is scored. Fetched assets must be **cached** so episodes are **replayable offline** deterministically.

Use `REPORT.md` to note trade-offs or shortcuts.

---

## **Deliverables**

1. **Python package** `geoguess_env/` (Py 3.10+)

   * `GeoGuessrEnv` (Gymnasium API).

   * Online sampler \+ local **cache manager**.

   * **Replay support** (freeze & offline replay).

   * `VLMBroker` that standardizes I/O for two actions: `click` and `answer`.

2. **Baseline agent** that follows arrows by clicking, then answers with a simple heuristic; writes a CSV of results.

3. **Tests** (pytest): determinism, **arrow click → movement** mapping, answer-only reward, replay fidelity, cache integrity.

4. **Docs**: `README.md` (install, keys, run) and `REPORT.md` (design/limits/next steps), plus `requirements.txt` or `pyproject.toml`.

---

## **Online sampling & caching**

* **Sample** `(lat, lon)` from a seeded RNG within a configurable **geofence** (avoid oceans).

* **Query** provider for nearest panorama; include **navigation links** (neighbors) if available.

* **Cache** all assets & metadata; later **replay offline** without network.

```
cache/
  images/
    <provider>_<pano_id>[_heading][_pitch].jpg
  metadata/
    manifest.jsonl   # one JSON per pano: request params, pano lat/lon, links, checksums, license
    attribution.md   # provider(s) & license notes
  replays/
    session_<seed>.json
```

*   
  **Networking occurs only in `reset()`** when `mode=online`. `step()` must be offline.

---

## **Environment specification**

### **API (Gymnasium)**

* `GeoGuessrEnv(config: dict)` — suggested keys:

  * `provider`, `mode ∈ {online, offline}`, `geofence`, `cache_root`

  * `obs_size` (e.g., `[512, 1024]` pano or a consistent perspective view)

  * `max_steps` (default 40), `seed`

  * `rate_limit_qps`, `max_fetch_retries`, `min_capture_year` (optional)

  * `arrow_hit_radius_px` (default 24), `arrow_min_conf` (default 0.0)

* `reset(seed: Optional[int] = None, options: Optional[dict] = None) -> (obs, info)`

* `step(action) -> (obs, reward, terminated, truncated, info)`

* `render(mode="rgb_array") -> np.ndarray`

* `close()`

### **Observation**

```py
obs = {
  "image": np.uint8[H, W, 3]   # current view visible to the agent
}
```

*   
  Use a consistent view per step (full equirectangular or a canonical perspective).

* You may overlay **no visual hints** (no arrow markers). Navigation targets must be **latent** for the agent—your env maps clicks to links internally.

### **Action space (two ops)**

1. **Click (navigation only; no reward)**

```json
{"op":"click","value":[x,y]}
```

   *   
     `x, y` are 0-indexed pixels in `obs["image"]`.

   * Env interprets the click as selecting a **navigation arrow** and moves to the corresponding neighbor pano **only if** the click lands sufficiently close to an arrow target (see *Arrow click semantics*).

   * If no arrow is selected, treat as **no-op** (stay in place) and continue.

2. **Answer (terminate & score)**

```json
{"op":"answer","value":[lat_deg,lon_deg]}
```

   *   
     Ends episode (`terminated=True`) and computes reward.

### **Arrow click semantics (required)**

Your env must convert pixels → **provider navigation links** as follows:

* **Link set**: For the current pano, maintain a list of **navigable links**  
   `links = [{id: str, heading_deg: float, rel_yaw_deg: float?, screen_xy: [cx, cy], conf: float?}, ...]`

  * If provider returns **explicit screen coordinates** / overlays for arrows, use them.

  * If not, **synthesize** a screen position per link from its **heading (and pitch if any)** using your camera model. A simple approach:

    * Assume the displayed view is centered at current yaw; project each neighbor’s `heading` to an `(cx, cy)` on the image (equirectangular: linear in x; keep a constant vertical band near horizon for `cy`).

    * Optionally store `conf=1.0` for synthesized points.

* **Hit test** (click → link):

  * Compute Euclidean distance in pixels between click `(x, y)` and each link center `(cx, cy)`.

  * Select the **nearest** link with `distance <= arrow_hit_radius_px` **and** `conf >= arrow_min_conf`.

  * **Tie-breakers** (in order): smallest distance → smallest absolute `rel_yaw_deg` (i.e., most forward) → lowest `pano_id` lexicographically.

  * If **no** link passes the threshold, the step is a **no-op** (return same pano).

* **Movement**:

  * If a link is selected, transition to that neighbor pano; update ground-truth `(lat, lon)`, current yaw (to the link’s heading), and refresh the link set for the new pano.

  * Count this as one step; **reward \= 0.0**.

* **Back/forward**:

  * If the provider exposes **back links**, they’re just normal links and are handled the same way.

  * You do **not** need a special “go back” action; the agent must click a back-pointing arrow.

### **Reward (answer-only)**

* Distance: Haversine with Earth radius 6371.0 km.

* Score \= Reward \= `exp(-distance_km / 400)`.

* Clicks always yield `0.0`.

* Termination:

  * `answer` ⇒ `terminated=True`

  * `max_steps` without answer ⇒ `truncated=True`, `reward=0.0`

### **`info` (include at least)**

```py
info = {
  "provider": str,
  "pano_id": str,
  "gt_lat": float, "gt_lon": float,
  "guess_lat": Optional[float], "guess_lon": Optional[float],
  "distance_km": Optional[float], "score": Optional[float],
  "steps": int,
  "pose": {"yaw_deg": float},       # current camera yaw
  "links": [                        # current navigable links (no screen hints to agent)
    {"id": str, "heading_deg": float, "screen_xy": [int,int], "conf": float}
  ]
}
```

---

## **VLMBroker (click \+ answer)**

* `build_prompt(image: np.ndarray, pose: dict) -> str`

  * Explain task and **two allowed actions**.

  * Require returning **only one JSON** object using this schema:

```json
{"op":"click","value":[x,y]}
```

  *   
    or

```json
{"op":"answer","value":[lat_deg,lon_deg]}
```

  *   
    May include *numeric* pose info (e.g., current yaw), but **no semantic hints**.

* `parse_action(text: str) -> dict`

  * Extract first JSON object; validate & clamp `click` to image bounds.

  * On error: return a safe **center click** (e.g., `{"op":"click","value":[W//2,H//2]}`) and a warning.

---

## **CLI**

**Online sample & cache while you go**

```shell
python -m geoguess_env.run_baseline \
  --mode online \
  --provider gsv \
  --episodes 30 \
  --geofence world_small.json \
  --cache ./cache \
  --seed 123 \
  --freeze-run ./cache/replays/session_123.json \
  --out results_online.csv
```

**Offline replay (deterministic grading)**

```shell
python -m geoguess_env.run_baseline \
  --mode offline \
  --replay ./cache/replays/session_123.json \
  --cache ./cache \
  --out results_offline.csv
```

**Replay file (example)**

```json
{
  "seed": 123,
  "episodes": [
    {
      "provider": "gsv",
      "pano_id": "abc123",
      "gt_lat": 40.6892,
      "gt_lon": -74.0445,
      "initial_yaw_deg": 0.0
    }
  ]
}
```

---

## **Baseline agent**

A minimal **arrow follower**:

* Clicks a small set of screen positions to sweep headings (e.g., thirds across width) for K steps or until it detects it’s looping (pano\_id repeats), then

* Emits an `answer` from a simple heuristic (e.g., sample from continent centroids or a global prior).

Outputs `results.csv`:

```
episode,pano_id,gt_lat,gt_lon,guess_lat,guess_lon,distance_km,score,steps
```

---

## **Testing (pytest)**

1. **Determinism & replay**

   * Same seed \+ geofence \+ provider ⇒ identical pre-fetch lat/lon samples.

   * Offline replay reproduces identical pano sequence and initial yaws.

2. **Arrow click mapping**

   * With a known link at `(cx,cy)`, clicks within `arrow_hit_radius_px` select that link; outside ⇒ no-op.

   * Tie-break logic respected.

   * After movement, `pano_id` changes and `pose.yaw_deg` aligns to chosen link heading (± wrap).

3. **Reward semantics**

   * Clicks: reward `0.0`.

   * `answer`: closer guess ⇒ strictly higher reward.

4. **Termination**

   * `answer` ⇒ `terminated=True`.

   * `max_steps` without `answer` ⇒ `truncated=True`, reward `0.0`.

5. **Cache integrity & offline safety**

   * SHA256 in `manifest.jsonl` matches file bytes.

   * Sockets disabled in offline mode; env continues to run.

6. **Observation**

   * `image` dtype `uint8`, shape matches config, values in `[0,255]`.

---

## **Provider notes & constraints**

* **APIs allowed:** Google Street View Static, Mapillary, KartaView, etc. Feel free to use your own keys.

* **Links/arrows:**

  * If provider returns link metadata (neighbors \+ headings), you **must** use it.

  * If it doesn’t return screen-space arrow polygons, **synthesize** arrow centers by projecting neighbor headings into the current image space (these centers are used only for hit-testing; do **not** draw them for the agent).

* **Attribution:** Include license text/links in `attribution.md`.

* **Rate limits:** Throttle QPS & backoff on 429/5xx.

* **Cache size target:** ≤500 MB.

* **Latency:** Fetch in `reset()` only; `step()` is offline and fast.

---

## **Submission**

* Repo/zip containing:

  * `geoguess_env/`, `tests/`, `run_baseline.py`

  * `README.md`, `REPORT.md`

  * `requirements.txt` / `pyproject.toml`

  * A tiny **sample cache** (5–10 images) to smoke-test offline