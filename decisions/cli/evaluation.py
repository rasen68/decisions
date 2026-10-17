"""Parse a restricted boolean expression over decision answer fields."""

import ast
import math
import operator
from typing import Any, Callable

from .api import DecisionError
from .questions import QuestionError

COMPARISONS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}
PRIMARY_FIELDS = {
    "predicate": "probability",
    "choice": "choice",
    "score": "score",
}


def outcome_probability(answer: dict[str, Any], outcome: str | bool) -> float:
    try:
        entries = answer["probabilities"]
        if not isinstance(entries, list):
            raise ValueError("expected a probabilities array")
        matches = [entry["probability"] for entry in entries
                   if type(entry["value"]) is type(outcome) and entry["value"] == outcome]
        if len(matches) != 1:
            raise ValueError("expected exactly one probability for the outcome")
        probability = matches[0]
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be a finite number between 0 and 1")
        return probability
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise DecisionError(f"invalid probability for choice outcome {outcome!r}: {error}") from error


def compile_expression(expression: str, questions: list[dict[str, Any]]) -> Callable[[list[dict[str, Any]]], bool]:
    """Validate syntax and question references before any request is sent."""
    references: dict[str, set[int]] = {}
    readers: list[Callable[[list[dict[str, Any]]], str | bool | int | float]] = []
    for index, question in enumerate(questions):
        references.setdefault(f"q{index + 1}", set()).add(index)
        if question.get("name"):
            references.setdefault(question["name"], set()).add(index)

    def invalid(message: str) -> None:
        raise QuestionError(f"invalid --eval expression: {message}")

    def literal(node: ast.AST) -> str | bool | int | float:
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name) and node.id in ("true", "false"):
            value = node.id == "true"
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = literal(node.operand)
            if type(value) not in (int, float):
                invalid("signs require a numeric literal")
            value = -value if isinstance(node.op, ast.USub) else value
        else:
            invalid("comparisons require a literal on the right")
        if type(value) not in (str, bool, int, float):
            invalid("expected a string, boolean, or number")
        if type(value) in (int, float):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                invalid("numbers must be finite")
        return value

    def compile_node(node: ast.AST) -> Callable:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            children = [compile_node(value) for value in node.values]
            combine = all if isinstance(node.op, ast.And) else any
            return lambda answers: combine(child(answers) for child in children)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            child = compile_node(node.operand)
            return lambda answers: not child(answers)
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            invalid("use comparisons joined by and, or, not, and parentheses")
        reference = node.left
        outcome = None
        if isinstance(reference, ast.Name):
            name, field = reference.id, None
        elif isinstance(reference, ast.Attribute) and isinstance(reference.value, ast.Name):
            name, field = reference.value.id, reference.attr
        elif isinstance(reference, ast.Subscript) and isinstance(reference.value, ast.Name):
            name, field = reference.value.id, "probability"
            outcome = literal(reference.slice)
            if type(outcome) not in (str, bool):
                invalid("choice outcomes must be string or boolean literals")
        else:
            invalid("the left side must be QUESTION, QUESTION.confidence, or QUESTION[OUTCOME]")
        indices = references.get(name, set())
        if len(indices) != 1:
            invalid(f"question reference {name!r} is {'ambiguous' if indices else 'unknown'}")
        index = next(iter(indices))
        kind = questions[index]["type"]
        if outcome is not None:
            if kind != "choice":
                invalid("outcome probabilities require a choice question")
            if not any(type(outcome) is type(entry["value"]) and outcome == entry["value"] for entry in questions[index]["choices"]):
                invalid(f"{outcome!r} is not a choice for {name}")
        elif field is None:
            field = PRIMARY_FIELDS[kind]
        elif field != "confidence" or kind == "predicate":
            invalid(f"{name}.{field} is not a field of a {kind} answer")
        comparison = COMPARISONS.get(type(node.ops[0]))
        if comparison is None:
            invalid("supported comparisons are ==, !=, <, <=, >, >=")
        value = literal(node.comparators[0])
        if field == "choice":
            if type(node.ops[0]) not in (ast.Eq, ast.NotEq) or type(value) not in (str, bool):
                invalid("choice comparisons require == or != and a string or boolean")
            if not any(type(value) is type(entry["value"]) and value == entry["value"] for entry in questions[index]["choices"]):
                invalid(f"{value!r} is not a choice for {name}")
        elif type(value) not in (int, float):
            label = f"{name}.confidence" if field == "confidence" else name
            invalid(f"{label} requires a numeric literal")
        slot = len(readers)
        if outcome is not None:
            readers.append(lambda answers: outcome_probability(answers[index], outcome))
        else:
            readers.append(lambda answers: answers[index][field])
        return lambda values: comparison(values[slot], value)

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise QuestionError(f"invalid --eval expression: {error}") from error
    try:
        evaluate = compile_node(tree.body)
    except RecursionError as error:
        raise QuestionError("invalid --eval expression: too deeply nested") from error
    # Resolve every comparison before boolean short-circuiting can hide bad data.
    return lambda answers: evaluate([read(answers) for read in readers])
