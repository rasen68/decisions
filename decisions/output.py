"""Format typed decision answers for a terminal."""

import json
import math
from typing import Any

from .api import DecisionError


def number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def percentage(value: Any) -> str:
    value = number(value)
    if not 0 <= value <= 1:
        raise ValueError("probability or confidence must be between 0 and 1")
    return f"{value:.1%}"


def choice_value(value: Any) -> str:
    if isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError("choice values must be strings or booleans")


def format_decision(response: dict[str, Any], questions: list[dict[str, Any]]) -> str:
    answers = response.get("answers")
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise DecisionError("OpenAI response must contain one answer per question")
    blocks = []
    for index, (question, answer) in enumerate(zip(questions, answers), 1):
        if not isinstance(answer, dict):
            raise DecisionError(f"OpenAI answer {index} must be an object")
        instructions = question["instructions"]
        name = question.get("name")
        heading = f"{index}. {name}: {instructions}" if name else f"{index}. {instructions}"
        lines = [heading]
        try:
            kind = answer["type"]
            if kind == "refusal":
                lines.append("  Refused")
            elif kind != question["type"]:
                raise ValueError("answer type does not match the question")
            elif kind == "predicate":
                lines.append(f"  Probability: {percentage(answer['probability'])}")
            else:
                if kind == "score":
                    score = number(answer["score"])
                    maximum = len(question["levels"]) - 1
                    if not 0 <= score <= maximum:
                        raise ValueError("score is outside the supplied levels")
                    lines.append(f"  Score: {score:.3f} / {maximum}")
                else:
                    lines.append(f"  Choice: {choice_value(answer['choice'])}")
                lines.append(f"  Confidence: {percentage(answer['confidence'])}")
                probabilities = answer["probabilities"]
                if not isinstance(probabilities, list) or not probabilities:
                    raise ValueError("expected a nonempty probabilities array")
                for entry in probabilities:
                    if kind == "score":
                        label = entry["label"]
                        if not isinstance(label, str):
                            raise ValueError("score labels must be strings")
                        value = number(entry["value"])
                        label = f"{value:g} ({label})"
                    else:
                        label = choice_value(entry["value"])
                    lines.append(f"    {label}: {percentage(entry['probability'])}")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise DecisionError(f"invalid OpenAI answer {index}: {error}") from error
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
