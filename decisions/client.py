"""Pooled synchronous and native asynchronous decision clients."""

from collections.abc import Mapping, Sequence
import os
from typing import Any, Self, cast

import httpx

from .providers.composition import PreparedRequest, PublicRequest, Target
from .providers.integrations import resolve
from .providers import OpenAI, Provider
from .types import (
    Choice, ChoiceAnswer, Choices, Image, Levels, Predicate, PredicateAnswer,
    Question, RefusalAnswer, Result, Score, ScoreAnswer, State,
    ConnectionError, HTTPError, InvalidResponseError, TimeoutError,
)


def _configuration(provider: Provider) -> tuple[str, dict[str, str]]:
    key = provider.api_key
    if key is None and provider.key_env is not None:
        key = os.environ.get(provider.key_env)
    # Basic key checks
    if key is None:
        if provider.key_env is not None:
            raise ValueError(f"{provider.name} requires an API key")
    elif not isinstance(key, str):
        raise TypeError("API key must be a string")
    elif not key.strip():
        raise ValueError("API key must not be blank")
    # Make headers
    headers = {"Accept": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    return provider.base_url.rstrip("/") + provider.path, headers


class Client:
    """A reusable client for one provider. Close it or use a with block."""

    def __init__(self,
                 provider: Provider | None = None,
                 *, timeout: float = 60.0):
        self._provider = provider if provider is not None else OpenAI()
        self._url, headers = _configuration(self.provider)
        self._target = Target(self.provider.name, self.provider.model, self._url, headers)
        self._http = httpx.Client(headers=headers, timeout=timeout)

    @property
    def provider(self) -> Provider:
        return self._provider

    def _send(self, body: dict[str, Any] | PreparedRequest[dict[str, Any]]) -> httpx.Response:
        if isinstance(body, PreparedRequest):
            return self._http.request(body.method, body.url, headers=body.headers, json=body.body)
        return self._http.post(self._url, json=body)

    def decide(self,
               questions: Mapping[str, Question],
               state: State | None = None,
               *, images: Sequence[Image] = ()
               ) -> Result:
        """ Core decide function, translate questions -> request -> result"""

        integration = resolve(self.provider)
        request = PublicRequest(questions, state, images)
        prepared = integration.prepare(request, self._target)
        try:
            response = self._send(prepared)

        # Basic error handling
        except httpx.TimeoutException as error:
            raise TimeoutError(f"{self.provider.name} request timed out") from error
        except httpx.RequestError as error:
            raise ConnectionError(f"{self.provider.name} request failed") from error
        if not response.is_success:
            raise HTTPError(response.status_code, self.provider.name, response.headers.get("x-request-id"))

        # Make sure we have valid JSON
        try:
            raw = response.json()
        except ValueError as error:
            raise InvalidResponseError(f"{self.provider.name} returned an invalid response") from error

        # Decode and return Result
        return integration.response.decode(request, self._target, raw)

    """ Simple single-question APIs"""
    def predicate(self,
                  question: str,
                  state: State | None = None,
                  *, images: Sequence[Image] = ()
                  ) -> PredicateAnswer | RefusalAnswer:
        return cast(PredicateAnswer | RefusalAnswer, self.decide({"answer": Predicate(question)}, state, images=images)["answer"])

    def choice(self,
               question: str,
               state: State | None = None,
               *, choices: Choices,
               images: Sequence[Image] = ()
               ) -> ChoiceAnswer | RefusalAnswer:
        return cast(ChoiceAnswer | RefusalAnswer, self.decide({"answer": Choice(question, choices=choices)}, state, images=images)["answer"])

    def score(self,
              question: str,
              state: State | None = None,
              *, levels: Levels,
              images: Sequence[Image] = ()) -> ScoreAnswer | RefusalAnswer:
        return cast(ScoreAnswer | RefusalAnswer, self.decide({"answer": Score(question, levels=levels)}, state, images=images)["answer"])

    # Alias function for noul
    noul = predicate

    """ Context manager helpers """
    def __enter__(self) -> Self:
        return self

    def close(self) -> None:
        self._http.close()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class AsyncClient:
    """An async client using the same wire conversion as Client."""

    def __init__(self,
                 provider: Provider | None = None,
                 *, timeout: float = 60.0):
        self._provider = provider if provider is not None else OpenAI()
        self._url, headers = _configuration(self.provider)
        self._target = Target(self.provider.name, self.provider.model, self._url, headers)
        self._http = httpx.AsyncClient(headers=headers, timeout=timeout)

    @property
    def provider(self) -> Provider:
        return self._provider

    async def _send(self, body: dict[str, Any] | PreparedRequest[dict[str, Any]]) -> httpx.Response:
        if isinstance(body, PreparedRequest):
            return await self._http.request(body.method, body.url, headers=body.headers, json=body.body)
        return await self._http.post(self._url, json=body)

    async def decide(self,
                     questions: Mapping[str, Question],
                     state: State | None = None,
                     *, images: Sequence[Image] = ()
                     ) -> Result:
        """ Core decide function, translate questions -> request -> result"""

        integration = resolve(self.provider)
        request = PublicRequest(questions, state, images)
        prepared = integration.prepare(request, self._target)
        try:
            response = await self._send(prepared)

        # Basic error handling
        except httpx.TimeoutException as error:
            raise TimeoutError(f"{self.provider.name} request timed out") from error
        except httpx.RequestError as error:
            raise ConnectionError(f"{self.provider.name} request failed") from error
        if not response.is_success:
            raise HTTPError(response.status_code, self.provider.name, response.headers.get("x-request-id"))

        # Make sure we have valid JSON
        try:
            raw = response.json()
        except ValueError as error:
            raise InvalidResponseError(f"{self.provider.name} returned an invalid response") from error

        # Decode and return Result
        return integration.response.decode(request, self._target, raw)

    """ Simple single-question APIs """
    async def predicate(self,
                        question: str,
                        state: State | None = None,
                        *, images: Sequence[Image] = ()
                        ) -> PredicateAnswer | RefusalAnswer:
        result = await self.decide({"answer": Predicate(question)}, state, images=images)
        return cast(PredicateAnswer | RefusalAnswer, result["answer"])

    async def choice(self,
                     question: str,
                     state: State | None = None,
                     *, choices: Choices,
                     images: Sequence[Image] = ()
                     ) -> ChoiceAnswer | RefusalAnswer:
        result = await self.decide({"answer": Choice(question, choices=choices)}, state, images=images)
        return cast(ChoiceAnswer | RefusalAnswer, result["answer"])

    async def score(self,
                    question: str,
                    state: State | None = None,
                    *, levels: Levels,
                    images: Sequence[Image] = ()
                    ) -> ScoreAnswer | RefusalAnswer:
        result = await self.decide({"answer": Score(question, levels=levels)}, state, images=images)
        return cast(ScoreAnswer | RefusalAnswer, result["answer"])

    # Alias function for noul
    noul = predicate

    """ Context manager helpers"""
    async def __aenter__(self) -> Self:
        return self

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()
