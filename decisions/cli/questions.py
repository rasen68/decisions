"""Convert CLI questions and validate the API question schema."""

from typing import Any


class QuestionError(ValueError):
    """A question does not follow the supported schema."""


def option(text: str, key: str) -> dict[str, str]:
    name, separator, description = text.partition(":")
    name = name.strip()
    description = description.strip() if separator else name
    if not name or not description:
        raise QuestionError("option names and descriptions must not be empty")
    return {key: name, "description": description}


def flag_question(kind: str, values: str | list[str]) -> dict[str, Any]:
    if isinstance(values, str):
        return {"type": kind, "instructions": values}
    if len(values) < 2:
        raise QuestionError(f"--{kind} needs a question followed by at least one option")
    question = {"type": kind, "instructions": values[0]}
    field, key = ("levels", "label") if kind == "score" else ("choices", "value")
    question[field] = [option(text, key) for text in values[1:]]
    return question


def validate_questions(questions: object) -> list[dict[str, Any]]:
    if not isinstance(questions, list) or not questions:
        raise QuestionError("questions must be a nonempty array")
    names: set[str] = set()
    for index, question in enumerate(questions, 1):
        where = f"question {index}"
        if not isinstance(question, dict):
            raise QuestionError(f"{where} must be an object")
        kind = question.get("type")
        if kind not in ("predicate", "choice", "score"):
            raise QuestionError(f"{where} type must be predicate, choice, or score")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise QuestionError(f"{where} needs nonempty instructions")
        name = question.get("name")
        if "name" in question:
            if not isinstance(name, str):
                raise QuestionError(f"{where} name must be a string")
            if name in names:
                raise QuestionError(f"duplicate question name: {name!r}")
            names.add(name)
        allowed = {"type", "instructions", "name"}
        if kind != "predicate":
            field, key = ("levels", "label") if kind == "score" else ("choices", "value")
            allowed.add(field)
            options = question.get(field)
            if not isinstance(options, list) or not options:
                raise QuestionError(f"{where} {field} must be a nonempty array")
            seen = set()
            for entry in options:
                if not isinstance(entry, dict) or key not in entry:
                    raise QuestionError(f"{where} {field} entries must be objects with {key}")
                value = entry[key]
                valid = isinstance(value, str) or (kind == "choice" and isinstance(value, bool))
                if not valid:
                    expected = "a string" if kind == "score" else "a string or boolean"
                    raise QuestionError(f"{where} {key} must be {expected}")
                identity = (type(value), value)
                if identity in seen:
                    raise QuestionError(f"{where} has duplicate {key}: {value!r}")
                seen.add(identity)
                if "description" in entry and not isinstance(entry["description"], str):
                    raise QuestionError(f"{where} descriptions must be strings")
                if entry.keys() - {key, "description"}:
                    raise QuestionError(f"{where} {field} entry has unsupported fields")
        extra = question.keys() - allowed
        if extra:
            raise QuestionError(f"{where} has unsupported fields: {', '.join(sorted(extra))}")
    return questions
