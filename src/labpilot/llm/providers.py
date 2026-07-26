"""Cloud LLM providers (OpenAI, Gemini).

Claude / OpenRouter are intentionally deferred — add them here when ready and
register in the router / client registry.
"""

from __future__ import annotations


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str) -> None:
        from openai import OpenAI  # optional dependency, see pyproject `llm` extra

        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str) -> None:
        from google import genai  # optional dependency, see pyproject `llm` extra
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._generate_content_config = types.GenerateContentConfig

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        temperature: float,
    ) -> str:
        response = self._client.models.generate_content(
            model=model,
            contents=user,
            config=self._generate_content_config(
                system_instruction=system,
                temperature=temperature,
            ),
        )
        return response.text or ""


class OpenAIClient:
    """Legacy LLMClient-shaped wrapper (model/temperature bound at construct)."""

    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        self._provider = OpenAIProvider(api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        return self._provider.complete(
            system,
            user,
            model=self.model,
            temperature=self.temperature,
        )


class GeminiClient:
    """Legacy LLMClient-shaped wrapper (model/temperature bound at construct)."""

    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        self._provider = GeminiProvider(api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        return self._provider.complete(
            system,
            user,
            model=self.model,
            temperature=self.temperature,
        )
