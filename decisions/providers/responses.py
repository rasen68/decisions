"""Validate provider responses and decode ordered public answers."""

from collections.abc import Mapping
from typing import Any

from .config import Provider
from ..types import (
    Answer, Choice, ChoiceAnswer, Predicate, PredicateAnswer,
    InvalidResponseError, Question, RefusalAnswer, Result, Score, ScoreAnswer, Usage,
)


def decode_choice_answer(question: Choice, answer: dict[str, Any], *, openai: bool) -> ChoiceAnswer:
    if openai:
        probabilities = {entry["value"]: entry["probability"] for entry in answer["probabilities"]}
    else:
        probabilities = answer["probabilities"]

    ordered_probabilities = {
        value: probabilities[value]
        for value in question.choices
    }

    return ChoiceAnswer(choice=answer["choice"], confidence=answer["confidence"], probabilities=ordered_probabilities)


def decode_score_answer(question: Score, answer: dict[str, Any], *, openai: bool) -> ScoreAnswer:
    labels = list(question.levels)

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
    expected = list(question.choices if isinstance(question, Choice) else question.levels)
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
            _resp_assert(any(type(choice) is type(option) and choice == option for option in question.choices))
            _validate_number(answer.get("confidence"))
            _validate_probabilities(question, answer, openai=openai)
        case Score():
            _resp_assert(answer.get("type") == "score")
            _validate_number(answer.get("score"), len(question.levels) - 1)
            _validate_number(answer.get("confidence"))
            _validate_probabilities(question, answer, openai=openai)


def _validate_response(questions: Mapping[str, Question], raw: Any, *, openai: bool) -> None:
    """ Validation of the contracts/invariants we expect from provider responses """
    # Response was a dict with fields we expect
    _resp_assert(isinstance(raw, dict))

    wire_answers = raw.get("answers")
    if openai:
        _resp_assert(isinstance(wire_answers, list))
        _resp_assert(len(wire_answers) == len(questions))
    else:
        _resp_assert(isinstance(wire_answers, dict))
        _resp_assert(all(name in wire_answers for name in questions))

    # Answers were dicts with fields we expect
    for index, name in enumerate(questions):
        answer = wire_answers[index] if openai else wire_answers[name]
        _resp_assert(isinstance(answer, dict))
        _validate_answer(questions[name], answer, openai=openai)

    _resp_assert(isinstance(raw.get("model"), str))

    usage = raw.get("usage")
    _resp_assert(isinstance(usage, dict))
    for field in ("input_tokens", "output_tokens"):
        value = usage.get(field)
        _resp_assert(isinstance(value, int) and not isinstance(value, bool))
        _resp_assert(value >= 0)


def convert_result(provider: str, questions: Mapping[str, Question], payload: dict[str, Any], raw: Any, *, openai: bool) -> Result:
    wire_answers = payload["answers"]
    answers = {}

    # OpenAI matches answers by position, Jev/Jeff use names
    for index, (name, question) in enumerate(questions.items()):
        answer = wire_answers[index] if openai else wire_answers[name]
        answers[name] = decode_answer(question, answer, openai=openai)

    usage = payload["usage"]

    return Result(
        answers=answers,
        provider=provider,
        model=payload["model"],
        usage=Usage(usage["input_tokens"], usage["output_tokens"]),
        raw=raw,
    )


def decode_result(provider: Provider, questions: Mapping[str, Question], raw: Any) -> Result:
    from .composition import PublicRequest, Target
    from .integrations import resolve

    target = Target(provider.name, provider.model, provider.base_url.rstrip("/") + provider.path)
    return resolve(provider).response.decode(PublicRequest(questions, None, ()), target, raw)
