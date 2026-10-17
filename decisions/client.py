"""Pooled synchronous and native asynchronous decision clients."""

from collections.abc import Mapping, Sequence
import os
from typing import Any, Self, cast

import httpx

from .providers.conversion import build_request, decode_result
from .providers import OpenAI, Provider
from .types import (
    Choice, ChoiceAnswer, Choices, Image, Levels, Predicate, PredicateAnswer,
    Question, Result, Score, ScoreAnswer, State,
)


def _configuration(provider: Provider) -> tuple[str, dict[str, str]]:
    key = provider.api_key
    if key is None and provider.key_env is not None:
        key = os.environ.get(provider.key_env)
    headers = {"Accept": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    return provider.base_url.rstrip("/") + provider.path, headers


class Client:
    """A reusable client for one provider. Close it or use a with block."""

    def __init__(self, provider: Provider | None = None, *, timeout: float = 60.0):
        self.provider = provider if provider is not None else OpenAI()
        self._url, headers = _configuration(self.provider)
        self._http = httpx.Client(headers=headers, timeout=timeout)

    def _send(self, body: dict[str, Any]) -> httpx.Response:
        return self._http.post(self._url, json=body)

    def decide(self,
               questions: Mapping[str, Question],
               state: State | None = None,
               *,
               images: Sequence[Image] = (),
               ) -> Result:
        response = self._send(build_request(self.provider, questions, state, images))
        return decode_result(self.provider, questions, response.json())

    def predicate(self, question: str, state: State | None = None, *, images: Sequence[Image] = ()) -> PredicateAnswer:
        return cast(PredicateAnswer, self.decide({"answer": Predicate(question)}, state, images=images)["answer"])

    noul = predicate

    def choice(self, question: str, state: State | None = None, *, choices: Choices, images: Sequence[Image] = ()) -> ChoiceAnswer:
        return cast(ChoiceAnswer, self.decide({"answer": Choice(question, choices=choices)}, state, images=images)["answer"])

    def score(self, question: str, state: State | None = None, *, levels: Levels, images: Sequence[Image] = ()) -> ScoreAnswer:
        return cast(ScoreAnswer, self.decide({"answer": Score(question, levels=levels)}, state, images=images)["answer"])

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class AsyncClient:
    """An async client using the same wire conversion as Client."""

    def __init__(self, provider: Provider | None = None, *, timeout: float = 60.0):
        self.provider = provider if provider is not None else OpenAI()
        self._url, headers = _configuration(self.provider)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)

    async def _send(self, body: dict[str, Any]) -> httpx.Response:
        return await self._http.post(self._url, json=body)

    async def decide(self, questions: Mapping[str, Question], state: State | None = None, *, images: Sequence[Image] = ()) -> Result:
        response = await self._send(build_request(self.provider, questions, state, images))
        return decode_result(self.provider, questions, response.json())

    async def predicate(self, question: str, state: State | None = None, *, images: Sequence[Image] = ()) -> PredicateAnswer:
        result = await self.decide({"answer": Predicate(question)}, state, images=images)
        return cast(PredicateAnswer, result["answer"])

    noul = predicate

    async def choice(self, question: str, state: State | None = None, *, choices: Choices, images: Sequence[Image] = ()) -> ChoiceAnswer:
        result = await self.decide({"answer": Choice(question, choices=choices)}, state, images=images)
        return cast(ChoiceAnswer, result["answer"])

    async def score(self, question: str, state: State | None = None, *, levels: Levels, images: Sequence[Image] = ()) -> ScoreAnswer:
        result = await self.decide({"answer": Score(question, levels=levels)}, state, images=images)
        return cast(ScoreAnswer, result["answer"])

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()
