"""Wire conversion shared by the synchronous and asynchronous clients."""

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from .config import OpenAI, Provider
from ..types import (
    Answer, Choice, ChoiceAnswer, Choices, Image, Levels, Predicate, PredicateAnswer,
    InvalidResponseError, Question, RefusalAnswer, Result, Score, ScoreAnswer, State, Usage,
)


def option_pairs(options: Choices | Levels) -> list[tuple[Any, str]]:
    if isinstance(options, Mapping):
        return list(options.items())
    return [(value, str(value)) for value in options]


def data_url(image: Image) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"


def _jeff_images(images: Sequence[Image]) -> dict[str, Any]:
    return {"images": [data_url(image) for image in images]}


_IMAGE_ENCODERS = {"jeff": _jeff_images}


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


def build_openai_input(state: State | None, images: Sequence[Image]) -> Any:
    if not images:
        return state

    parts = []
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
            wire["type"] = "choice"
            wire["criteria"] = dict(option_pairs(question.choices))
        case Score():
            wire["type"] = "score"
            wire["criteria"] = [d for _, d in option_pairs(question.levels)]

    return wire


def build_request(provider: Provider, questions: Mapping[str, Question], state: State | None, images: Sequence[Image]) -> dict[str, Any]:
    if isinstance(provider, OpenAI):
        return {
            "model": provider.model,
            "input": build_openai_input(state, images),
            "questions": [
                build_openai_question(name, question)
                for name, question in questions.items()
            ],
        }

    request = {
        "model": provider.model,
        "state": "" if state is None else state,
        "questions": {
            name: build_systemone_question(question)
            for name, question in questions.items()
        },
    }
    if images:
        request.update(_IMAGE_ENCODERS[provider.name](images))
    return request


# Response conversion into public values and request ordering.


def decode_choice_answer(question: Choice, answer: dict[str, Any], *, openai: bool) -> ChoiceAnswer:
    if openai:
        probabilities = {entry["value"]: entry["probability"] for entry in answer["probabilities"]}
    else:
        probabilities = answer["probabilities"]

    ordered_probabilities = {
        value: probabilities[value]
        for value, _ in option_pairs(question.choices)
    }

    return ChoiceAnswer(choice=answer["choice"], confidence=answer["confidence"], probabilities=ordered_probabilities)


def decode_score_answer(question: Score, answer: dict[str, Any], *, openai: bool) -> ScoreAnswer:
    labels = [label for label, _ in option_pairs(question.levels)]

    if openai:
        probabilities = {
            entry["label"]: entry["probability"]
            for entry in answer["probabilities"]
        }
    else:
        probabilities = {
            label: answer["probabilities"][str(index)]
            for index, label in enumerate(labels)
        }

    ordered_probabilities = {label: probabilities[label] for label in labels}

    return ScoreAnswer(
        score=answer["score"],
        confidence=answer["confidence"],
        probabilities=ordered_probabilities,
    )


def decode_answer(question: Question, answer: dict[str, Any], *, openai: bool) -> Answer:
    if openai and answer["type"] == "refusal":
        return RefusalAnswer()
    match question:
        case Predicate():
            field = "probability" if openai else "noul"
            return PredicateAnswer(answer[field])
        case Choice():
            return decode_choice_answer(question, answer, openai=openai)
        case Score():
            return decode_score_answer(question, answer, openai=openai)


def _resp_assert(condition: bool):
    if not condition:
        raise InvalidResponseError

def _validate_number(value: Any, maximum: int = 1) -> None:
    _resp_assert(isinstance(value, (int, float)) and not isinstance(value, bool))
    _resp_assert(0 <= value <= maximum)


def _validate_probabilities(question: Choice | Score, answer: dict[str, Any], *, openai: bool) -> None:
    options = option_pairs(question.choices if isinstance(question, Choice) else question.levels)
    expected = [value for value, _ in options]
    probabilities = answer.get("probabilities")

    if openai:
        _resp_assert(isinstance(probabilities, list))
        _resp_assert(len(probabilities) == len(expected))
        seen = set()
        for entry in probabilities:
            _resp_assert(isinstance(entry, dict))
            if isinstance(question, Choice):
                value = entry.get("value")
                _resp_assert(type(value) in (str, bool))
                key = (type(value), value)
                _resp_assert(any(type(value) is type(option) and value == option for option in expected))
            else:
                value = entry.get("label")
                _resp_assert(isinstance(value, str) and value in expected)
                index = entry.get("value")
                _resp_assert(isinstance(index, int) and not isinstance(index, bool))
                _resp_assert(index == expected.index(value))
                key = value
            _resp_assert(key not in seen)
            seen.add(key)
            _validate_number(entry.get("probability"))
    else:
        _resp_assert(isinstance(probabilities, dict))
        keys = expected if isinstance(question, Choice) else [str(index) for index in range(len(expected))]
        _resp_assert(set(probabilities) == set(keys))
        for value in probabilities.values():
            _validate_number(value)


def _validate_answer(question: Question, answer: dict[str, Any], *, openai: bool) -> None:
    if openai and answer.get("type") == "refusal":
        _resp_assert("name" in answer)
        _resp_assert(answer["name"] is None or isinstance(answer["name"], str))
        return

    match question:
        case Predicate():
            _resp_assert(answer.get("type") == ("predicate" if openai else "noul"))
            _validate_number(answer.get("probability" if openai else "noul"))
        case Choice():
            _resp_assert(answer.get("type") == "choice")
            choice = answer.get("choice")
            _resp_assert(any(type(choice) is type(option) and choice == option for option, _ in option_pairs(question.choices)))
            _validate_number(answer.get("confidence"))
            _validate_probabilities(question, answer, openai=openai)
        case Score():
            _resp_assert(answer.get("type") == "score")
            _validate_number(answer.get("score"), len(question.levels) - 1)
            _validate_number(answer.get("confidence"))
            _validate_probabilities(question, answer, openai=openai)


def _validate_response(provider: Provider, questions: Mapping[str, Question], raw: Any) -> None:
    """ Validation of the contracts/invariants we expect from provider responses """
    # Response was a dict with fields we expect
    _resp_assert(isinstance(raw, dict))

    wire_answers = raw.get("answers")
    if isinstance(provider, OpenAI):
        _resp_assert(isinstance(wire_answers, list))
        _resp_assert(len(wire_answers) == len(questions))
    else:
        _resp_assert(isinstance(wire_answers, dict))
        _resp_assert(all(name in wire_answers for name in questions))

    # Answers were dicts with fields we expect
    for index, name in enumerate(questions):
        answer = wire_answers[index] if isinstance(provider, OpenAI) else wire_answers[name]
        _resp_assert(isinstance(answer, dict))
        _validate_answer(questions[name], answer, openai=isinstance(provider, OpenAI))

    _resp_assert(isinstance(raw.get("model"), str))

    usage = raw.get("usage")
    _resp_assert(isinstance(usage, dict))
    for field in ("input_tokens", "output_tokens"):
        value = usage.get(field)
        _resp_assert(isinstance(value, int) and not isinstance(value, bool))
        _resp_assert(value >= 0)


def decode_result(provider: Provider, questions: Mapping[str, Question], raw: Any) -> Result:
    _validate_response(provider, questions, raw)
    openai = isinstance(provider, OpenAI)
    wire_answers = raw["answers"]
    answers = {}

    # OpenAI matches answers by position, Jev/Jeff use names
    for index, (name, question) in enumerate(questions.items()):
        answer = wire_answers[index] if openai else wire_answers[name]
        answers[name] = decode_answer(question, answer, openai=openai)

    usage = raw["usage"]

    return Result(
        answers=answers,
        provider=provider.name,
        model=raw["model"],
        usage=Usage(usage["input_tokens"], usage["output_tokens"]),
        raw=raw,
    )
