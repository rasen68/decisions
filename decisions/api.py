"""A minimal standard-library HTTP client for OpenAI Decisions."""

import json
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

ENDPOINT = "https://api.openai.com/v1/decisions"
TIMEOUT = 60


class DecisionError(Exception):
    """An API request or response could not be used."""


def request_decision(input_data: str | list[dict[str, Any]], questions: list[dict[str, Any]], model: str, api_key: str) -> str:
    body = json.dumps({"model": model, "input": input_data, "questions": questions}, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        try:
            with error:
                message = error.read().decode("utf-8", errors="replace").strip()
        except (HTTPException, OSError) as body_error:
            raise DecisionError(f"OpenAI HTTP {error.code}: could not read error response: {body_error}") from body_error
        try:
            detail = json.loads(message).get("error")
            if isinstance(detail, dict):
                message = str(detail.get("message", message))
            elif isinstance(detail, str):
                message = detail
        except (ValueError, AttributeError):
            pass
        raise DecisionError(f"OpenAI HTTP {error.code}: {message[:1000] or error.reason}") from error
    except URLError as error:
        raise DecisionError(f"could not reach OpenAI: {error.reason}") from error
    except TimeoutError as error:
        raise DecisionError(f"OpenAI request timed out after {TIMEOUT} seconds") from error
    except HTTPException as error:
        raise DecisionError(f"invalid HTTP response from OpenAI: {error}") from error
    except UnicodeDecodeError as error:
        raise DecisionError("OpenAI returned a response that is not UTF-8") from error
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise DecisionError("OpenAI returned invalid JSON") from error
    if not isinstance(data, dict):
        raise DecisionError("OpenAI response must be a JSON object")
    return raw
