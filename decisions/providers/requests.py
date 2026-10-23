"""Build provider requests from public questions, state and images."""

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from .config import Provider
from ..types import Choice, Choices, Image, Levels, Predicate, Question, Score, State


def option_pairs(options: Choices | Levels) -> list[tuple[Any, str]]:
    if isinstance(options, Mapping):
        return list(options.items())
    return [(value, str(value)) for value in options]


def data_url(image: Image) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def build_openai_question(name: str, question: Question) -> dict[str, Any]:
    wire: dict[str, Any] = {"name": name, "instructions": question.question}
    match question:
        case Predicate():
            wire["type"] = "predicate"
        case Choice():
            wire["type"] = "choice"
            wire["choices"] = [{"value": v, "description": d} for v, d in option_pairs(question.choices)]
        case Score():
            wire["type"] = "score"
            wire["levels"] = [{"label": l, "description": d} for l, d in option_pairs(question.levels)]
    return wire


def build_openai_input(state: State | None, images: Sequence[Image]) -> str | list[dict[str, Any]] | None:
    if state is not None and not isinstance(state, str):
        raise ValueError("OpenAI content encoding requires text state")
    if not images:
        return state

    parts: list[dict[str, Any]] = []
    if state is not None:
        parts.append({"type": "input_text", "text": state})

    parts.extend(
        {"type": "input_image", "image_url": data_url(image)}
        for image in images
    )

    return [{"role": "user", "content": parts}]


# TypeSafe Jev / Jeff
def build_systemone_question(question: Question) -> dict[str, Any]:
    wire: dict[str, Any] = {"instructions": question.question}

    match question:
        case Predicate():
            wire["type"] = "noul"
        case Choice():
            if any(not isinstance(value, str) for value in question.choices):
                raise ValueError("System One criteria encoding requires string keys")
            wire["type"] = "choice"
            wire["criteria"] = dict(option_pairs(question.choices))
        case Score():
            wire["type"] = "score"
            wire["criteria"] = [d for _, d in option_pairs(question.levels)]

    return wire


def build_request(provider: Provider, questions: Mapping[str, Question], state: State | None, images: Sequence[Image]) -> dict[str, Any]:
    from .composition import PublicRequest, Target
    from .integrations import resolve

    target = Target(provider.name, provider.model, provider.base_url.rstrip("/") + provider.path)
    return resolve(provider).prepare(PublicRequest(questions, state, images), target).body
