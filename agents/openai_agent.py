from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from agents.base import AgentConfig, BaseAgent
from agents.openai_models import AnswerParams, ClickParams
from geoguess_env.vlm_broker import VLMBroker

from .utils import (
    cache_get,
    cache_put,
    compute_image_hash,
    compute_prompt_fingerprint,
    encode_image_to_jpeg_base64,
)


class OpenAIVisionAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config)
        # Lazy import to avoid hard dep if not used
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "openai package is required for OpenAIVisionAgent. Install 'openai'."
            ) from e
        # Load .env if available
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        # Broker for standardized prompt and parsing
        self._broker = VLMBroker()

    def reset(self) -> None:
        return None

    def act(self, observation, info: Dict[str, Any]) -> Dict[str, Any]:
        context = self._prepare_context(observation, info)

        cached_response = self._get_cached_response(context)
        if cached_response:
            return self._parse_action_or_fallback(
                cached_response,
                context["links"],
                image_shape=context["image_shape"],
            )

        messages, user_text = self._build_messages(context)
        print("USER TEXT: ", user_text, "\n")

        tools, tool_choice = self._decide_tools(context["force_answer"])
        response = self._chat_completions(messages, tools, tool_choice)
        action = self._parse_action_or_fallback(
            response,
            context["links"],
            image_shape=context["image_shape"],
        )
        print("ACTION: ", action, "\n")

        self._store_cached_response(context, response)

        return action

    # --- OpenAI call with retries ---
    def _prepare_context(self, observation, info: Dict[str, Any]) -> Dict[str, Any]:
        links = info.get("links", []) or []
        pose = info.get("pose") or {}
        heading_deg = pose.get("yaw_deg")
        if heading_deg is None:
            heading_deg = pose.get("heading_deg")

        raw_steps = info.get("steps")
        try:
            steps_taken = int(raw_steps)
        except (TypeError, ValueError):
            steps_taken = 0

        meta = {
            "pano_id": info.get("pano_id"),
            "steps": raw_steps,
            "heading_deg": heading_deg,
            "max_steps": int(self.config.max_nav_steps),
        }

        image = (
            observation if hasattr(observation, "shape") else observation.get("image")
        )
        if image is None:
            raise ValueError(
                "Observation does not contain an image for OpenAIVisionAgent"
            )

        image_shape = getattr(image, "shape", None)

        fingerprint = None
        if self.config.cache_dir:
            image_hash = compute_image_hash(image)
            fingerprint = compute_prompt_fingerprint(image_hash, links, meta)

        force_answer = meta["max_steps"] - steps_taken <= 3

        return {
            "image": image,
            "image_shape": image_shape,
            "links": links,
            "meta": meta,
            "fingerprint": fingerprint,
            "force_answer": force_answer,
            "steps_taken": steps_taken,
        }

    def _get_cached_response(self, context: Dict[str, Any]) -> Dict[str, Any] | None:
        fingerprint = context.get("fingerprint")
        if not (self.config.cache_dir and fingerprint):
            return None
        return cache_get(self.config.cache_dir, fingerprint)

    def _store_cached_response(
        self, context: Dict[str, Any], response: Dict[str, Any]
    ) -> None:
        fingerprint = context.get("fingerprint")
        if not (self.config.cache_dir and fingerprint and response):
            return
        cache_put(self.config.cache_dir, fingerprint, response)

    def _build_messages(
        self, context: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], str]:
        image = context["image"]
        meta = context["meta"]
        force_answer = context["force_answer"]
        steps_taken = context.get("steps_taken", 0)

        b64_image = encode_image_to_jpeg_base64(image, quality=95)
        broker_prompt = self._broker.build_prompt(
            image,
            {"yaw_deg": meta.get("heading_deg")},
        )

        tool_instructions = (
            "You MUST respond by calling exactly one tool: 'click' or 'answer'. "
            "Never output free-form text or JSON in your message; only call the tool."
        )
        if force_answer:
            tool_instructions += " You MUST call the 'answer' tool now."

        system_prompt = "You are assisting with GeoGuessr navigation. Use structured tool calls only."
        user_text = (
            f"pano_id={meta['pano_id']} steps={steps_taken}/{meta['max_steps']} heading_deg={meta['heading_deg']}\n\n"
            f"{broker_prompt}\n\n"
            f"{tool_instructions}"
        )

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

        return messages, user_text

    def _decide_tools(
        self, force_answer: bool
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | str]:
        tools = self._tools_schema(force_answer=force_answer)
        tool_choice: Dict[str, Any] | str = (
            {"type": "function", "function": {"name": "answer"}}
            if force_answer
            else "auto"
        )
        return tools, tool_choice

    def _chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        tool_choice: Dict[str, Any] | str | None,
    ) -> Dict[str, Any]:
        # Use tool calling to obtain structured action arguments; temperature from config
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    tools=tools or self._tools_schema(force_answer=False),
                    tool_choice=tool_choice if tool_choice is not None else "auto",
                    timeout=self.config.request_timeout_s,
                )
                msg = result.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []
                for call in tool_calls:
                    try:
                        fn = getattr(call, "function", None) or {}
                        name = (
                            getattr(fn, "name", None)
                            if not isinstance(fn, dict)
                            else fn.get("name")
                        )
                        args = (
                            getattr(fn, "arguments", None)
                            if not isinstance(fn, dict)
                            else fn.get("arguments")
                        )
                        if name == "click" and args is not None:
                            if isinstance(args, str):
                                model = ClickParams.model_validate_json(args)
                            else:
                                model = ClickParams.model_validate(args)
                            return {
                                "op": "click",
                                "click": {"x": model.x, "y": model.y},
                            }
                        if name == "answer" and args is not None:
                            if isinstance(args, str):
                                model = AnswerParams.model_validate_json(args)
                            else:
                                model = AnswerParams.model_validate(args)
                            return {
                                "op": "answer",
                                "answer": {"lat": model.lat, "lon": model.lon},
                            }
                    except Exception:
                        # Try next tool call if parsing one fails
                        continue
                # If no tool call returned the expected function, attempt to extract text
                try:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str) and content.strip():
                        return {"raw_text": content}
                    if isinstance(content, list):
                        texts = [
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ]
                        text_joined = "\n".join(t for t in texts if t)
                        if text_joined.strip():
                            return {"raw_text": text_joined}
                except Exception:
                    pass
            except Exception:
                time.sleep(0.8 * (2**attempt))
        # As a fallback, return empty dict to trigger heuristic
        return {}

    # --- Parsing and fallback ---
    def _parse_action_or_fallback(
        self,
        data: Dict[str, Any],
        links: List[Dict[str, Any]],
        image_shape: Any | None = None,
    ) -> Dict[str, Any]:
        try:
            op = data.get("op")
            if op == "click":
                click = data.get("click") or {}
                x = int(click.get("x"))
                y = int(click.get("y"))
                # Snap to nearest provided link center to guarantee hit
                x, y = self._snap_to_nearest_link(x, y, links)
                return {"op": "click", "value": [x, y]}
            if op == "answer":
                ans = data.get("answer") or {}
                lat = float(ans.get("lat"))
                lon = float(ans.get("lon"))
                lat = max(-90.0, min(90.0, lat))
                lon = max(-180.0, min(180.0, lon))
                return {"op": "answer", "value": [lat, lon]}
            # If the model returned raw text (no tool call), parse via broker
            raw_text = data.get("raw_text")
            if isinstance(raw_text, str) and raw_text.strip():
                h = self.config.image_height
                w = self.config.image_width
                if image_shape and len(image_shape) >= 2:
                    h = int(image_shape[0])
                    w = int(image_shape[1])
                parsed = self._broker.parse_action(
                    raw_text, image_width=w, image_height=h
                )
                if parsed.get("op") == "click":
                    vx, vy = parsed.get("value", [w // 2, h // 2])
                    sx, sy = self._snap_to_nearest_link(int(vx), int(vy), links)
                    return {"op": "click", "value": [sx, sy]}
                if parsed.get("op") == "answer":
                    vlat, vlon = parsed.get("value", [0.0, 0.0])
                    vlat = max(-90.0, min(90.0, float(vlat)))
                    vlon = max(-180.0, min(180.0, float(vlon)))
                    return {"op": "answer", "value": [vlat, vlon]}
        except Exception:
            pass
        # Heuristic: if any links exist, click the first one's center; else answer (0,0)
        if links:
            sx, sy = links[0].get("screen_xy", [512, 256])
            return {"op": "click", "value": [int(sx), int(sy)]}
        return {"op": "answer", "value": [0.0, 0.0]}

    def _tools_schema(self, force_answer: bool = False) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        if not force_answer:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "click",
                        "description": (
                            "Click a navigational link by specifying its screen coordinates."
                        ),
                        "parameters": ClickParams.model_json_schema(),
                    },
                }
            )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "answer",
                    "description": "Submit a final answer as latitude and longitude.",
                    "parameters": AnswerParams.model_json_schema(),
                },
            }
        )
        return tools

    @staticmethod
    def _snap_to_nearest_link(
        x: int, y: int, links: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        best = None
        best_d2 = None
        for link in links:
            sx, sy = link.get("screen_xy", [x, y])
            dx = int(x) - int(sx)
            dy = int(y) - int(sy)
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (int(sx), int(sy))
        return best if best is not None else (int(x), int(y))
