from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from .base import BaseAgent, AgentConfig
from .utils import encode_image_to_jpeg_base64, compute_image_hash, compute_prompt_fingerprint, cache_get, cache_put


class OpenAIVisionAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        # Lazy import to avoid hard dep if not used
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package is required for OpenAIVisionAgent. Install 'openai'.") from e
        # Load .env if available
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def reset(self) -> None:
        return None

    def act(self, observation, info: Dict[str, Any]) -> Dict[str, Any]:
        # Prepare inputs
        links = info.get("links", []) or []
        meta = {
            "pano_id": info.get("pano_id"),
            "steps": info.get("steps"),
            "heading_deg": (info.get("pose") or {}).get("heading_deg"),
            "max_steps": int(self.config.max_nav_steps),
        }

        # Optional caching
        image_hash = compute_image_hash(observation)
        fingerprint = compute_prompt_fingerprint(image_hash, links, meta)
        cached = cache_get(self.config.cache_dir, fingerprint) if self.config.cache_dir else None
        if cached:
            action = self._parse_action_or_fallback(cached, links)
            return action

        # Build messages
        b64_image = encode_image_to_jpeg_base64(observation, quality=95)
        system_prompt = (
            "You are navigating Street View-like panoramas. You can either click a link to move "
            "(by outputting the exact provided screen_xy) or answer with final latitude/longitude. "
            "If you have a good guess, always return an answer."
            "Output ONLY a JSON object matching this schema: {\"op\": \"click\"|\"answer\", \"click\": {\"x\": int, \"y\": int}, \"answer\": {\"lat\": float, \"lon\": float}}. "
            "Do not include any additional keys or text."
        )

        link_list_str = json.dumps([
            {"id": l.get("id"), "heading_deg": l.get("heading_deg"), "screen_xy": l.get("screen_xy")}
            for l in links
        ])
        force_answer = meta["max_steps"] - meta["steps"] <= 3
        user_text = (
            f"pano_id={meta['pano_id']} **steps={meta['steps']}** max_steps={meta['max_steps']} heading_deg={meta['heading_deg']}\n"
            f"links={link_list_str}\n"
            "Rules: To move, choose a link and click exactly its screen_xy center."
            "Answer with lat/lon for final answer. "
        )
        if force_answer:
            user_text += "You MUST return answer this time!!!"
        print("USER TEXT: ", user_text, "\n")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            },
        ]

        response = self._chat_completions(messages)
        action = self._parse_action_or_fallback(response, links)
        print("ACTION: ", action, "\n")

        if self.config.cache_dir:
            cache_put(self.config.cache_dir, fingerprint, response)

        return action

    # --- OpenAI call with retries ---
    def _chat_completions(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Using JSON-like constraint by post-parse; temperature=0 for determinism
        max_attempts = 3
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                result = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                    timeout=self.config.request_timeout_s,
                )
                content = result.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                last_error = e
                time.sleep(0.8 * (2 ** attempt))
        # As a fallback, return empty dict to trigger heuristic
        return {}

    # --- Parsing and fallback ---
    def _parse_action_or_fallback(self, data: Dict[str, Any], links: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            op = data.get("op")
            if op == "click":
                click = data.get("click") or {}
                x = int(click.get("x"))
                y = int(click.get("y"))
                # Snap to nearest provided link center to guarantee hit
                x, y = self._snap_to_nearest_link(x, y, links)
                return {"op": "click", "click": [x, y]}
            if op == "answer":
                ans = data.get("answer") or {}
                lat = float(ans.get("lat"))
                lon = float(ans.get("lon"))
                lat = max(-90.0, min(90.0, lat))
                lon = max(-180.0, min(180.0, lon))
                return {"op": "answer", "answer": [lat, lon]}
        except Exception:
            pass
        # Heuristic: if any links exist, click the first one's center; else answer (0,0)
        if links:
            sx, sy = links[0].get("screen_xy", [512, 256])
            return {"op": "click", "click": [int(sx), int(sy)]}
        return {"op": "answer", "answer": [0.0, 0.0]}

    @staticmethod
    def _snap_to_nearest_link(x: int, y: int, links: List[Dict[str, Any]]) -> tuple[int, int]:
        best = None
        best_d2 = None
        for l in links:
            sx, sy = l.get("screen_xy", [x, y])
            dx = int(x) - int(sx)
            dy = int(y) - int(sy)
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (int(sx), int(sy))
        return best if best is not None else (int(x), int(y))


