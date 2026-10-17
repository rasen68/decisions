"""CLI response and error policy over the library's OpenAI transport."""

import json
from typing import Any

import httpx

from ..client import Client
from ..providers import OpenAI

TIMEOUT = 60


class DecisionError(Exception):
    """An API request or response could not be used."""


def request_decision(input_data: str | list[dict[str, Any]], questions: list[dict[str, Any]], model: str, api_key: str) -> str:
    body = {"model": model, "input": input_data, "questions": questions}
    try:
        with Client(OpenAI(api_key=api_key, model=model), timeout=TIMEOUT) as client:
            response = client._send(body)
    except httpx.TimeoutException as error:
        raise DecisionError(f"OpenAI request timed out after {TIMEOUT} seconds") from error
    except httpx.RemoteProtocolError as error:
        raise DecisionError(f"invalid HTTP response from OpenAI: {error}") from error
    except httpx.RequestError as error:
        raise DecisionError(f"could not reach OpenAI: {error}") from error
    if response.is_error:
        message = response.content.decode("utf-8", errors="replace").strip()
        try:
            detail = json.loads(message).get("error")
            if isinstance(detail, dict):
                message = str(detail.get("message", message))
            elif isinstance(detail, str):
                message = detail
        except (ValueError, AttributeError):
            pass
        raise DecisionError(f"OpenAI HTTP {response.status_code}: {message[:1000] or response.reason_phrase}")
    try:
        raw = response.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DecisionError("OpenAI returned a response that is not UTF-8") from error
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise DecisionError("OpenAI returned invalid JSON") from error
    if not isinstance(data, dict):
        raise DecisionError("OpenAI response must be a JSON object")
    return raw
