from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI, OpenAIError, RateLimitError


class OpenRouterClient:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        # Determine provider: explicitly from LLM_PROVIDER or inferred from keys
        self.provider = os.getenv("LLM_PROVIDER")
        if not self.provider:
            if os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
            else:
                self.provider = "openrouter"

        self.timeout_seconds = float(
            os.getenv("LLM_TIMEOUT_SECONDS") or os.getenv("OPENROUTER_TIMEOUT_SECONDS", "45")
        )

        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini")
            self.base_url = "https://openrouter.ai/api/v1"

        self.enabled = bool(self.api_key)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.client = (
            OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                default_headers=(
                    {
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "Meeting Scheduling Agent",
                    }
                    if self.provider == "openrouter"
                    else None
                ),
            )
            if self.enabled
            else None
        )

    def _track_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            p_tokens = getattr(usage, "prompt_tokens", 0) or 0
            c_tokens = getattr(usage, "completion_tokens", 0) or 0
            cost = self._calculate_cost(self.model, p_tokens, c_tokens)
            self.total_prompt_tokens += p_tokens
            self.total_completion_tokens += c_tokens
            self.total_cost += cost

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        model_lower = model.lower()
        if "free" in model_lower:
            return 0.0
        
        # Pricing per 1M tokens in USD
        input_rate = 0.15  # Default to gpt-4o-mini / gpt-5-mini
        output_rate = 0.60
        
        if "gemini" in model_lower:
            input_rate = 0.075
            output_rate = 0.30
        elif "gpt-3.5" in model_lower:
            input_rate = 0.50
            output_rate = 1.50
        elif "gpt-4" in model_lower and "mini" not in model_lower:
            input_rate = 2.50
            output_rate = 10.00
            
        cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000.0
        return cost

    def extract_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.client:
            key_name = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENROUTER_API_KEY"
            raise RuntimeError(f"{key_name} is not set")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except RateLimitError as exc:
            raise RuntimeError(f"Rate limited by provider: {exc}") from exc
        except OpenAIError as exc:
            first_error = exc
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        messages[0],
                        {"role": "user", "content": user + "\n\nReturn only valid JSON. No markdown."},
                    ],
                    temperature=0,
                )
            except RateLimitError as exc:
                raise RuntimeError(f"Rate limited by provider: {exc}") from exc
            except OpenAIError as exc:
                raise RuntimeError(f"{self.provider.capitalize()} request failed: {exc}") from first_error
        self._track_usage(response)
        if not hasattr(response, "choices") or not response.choices:
            raise RuntimeError(f"LLM API returned an empty or invalid response without choices. Response: {response}")
        content = response.choices[0].message.content or "{}"
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
        return json.loads(content)

    def generate_text(self, system: str, user: str, temperature: float = 0.3) -> str:
        if not self.client:
            key_name = "GEMINI_API_KEY" if self.provider == "gemini" else "OPENROUTER_API_KEY"
            raise RuntimeError(f"{key_name} is not set")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
        except RateLimitError as exc:
            raise RuntimeError(f"Rate limited by provider: {exc}") from exc
        except OpenAIError as exc:
            raise RuntimeError(f"{self.provider.capitalize()} request failed: {exc}") from exc
        self._track_usage(response)
        if not hasattr(response, "choices") or not response.choices:
            raise RuntimeError(f"LLM API returned an empty or invalid response without choices. Response: {response}")
        return (response.choices[0].message.content or "").strip()
