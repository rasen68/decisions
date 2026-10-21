"""Reusable provider-independent question definitions."""

from collections.abc import Mapping
from dataclasses import dataclass, field

ChoiceValue = str | bool
# Lists and mapping keys are invariant, so include homogeneous choices explicitly.
Choices = (
    list[str] | list[bool] | list[ChoiceValue]
    | Mapping[str, str] | Mapping[bool, str] | Mapping[ChoiceValue, str]
)
Levels = list[str] | Mapping[str, str]

@dataclass
class Predicate:
    question: str

    def __post_init__(self) -> None:
        validate_question(self)

Noul = Predicate

@dataclass
class Choice:
    question: str
    choices: Choices = field(kw_only=True)

    def __post_init__(self) -> None:
        validate_question(self)

@dataclass
class Score:
    question: str
    levels: Levels = field(kw_only=True)

    def __post_init__(self) -> None:
        validate_question(self)

Question = Predicate | Choice | Score


def _validate_text(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("Question text and names must be strings")
    if not value.strip():
        raise ValueError("Question text and names must not be blank")


def _validate_options(options: Choices | Levels, *, allow_booleans: bool) -> None:
    if not isinstance(options, (list, Mapping)):
        raise TypeError("Options must be a list or mapping")
    if not options:
        raise ValueError("Options must not be empty")

    seen = set()
    for value in options:
        if not isinstance(value, str) and not (allow_booleans and isinstance(value, bool)):
            raise TypeError("Option values must be strings or supported booleans")
        key = (type(value), value)
        if key in seen:
            raise ValueError("Options must not contain duplicates")
        seen.add(key)

    if isinstance(options, Mapping):
        for description in options.values():
            if not isinstance(description, str):
                raise TypeError("Option descriptions must be strings")


def validate_question(question: Question) -> None:
    if not isinstance(question, (Predicate, Choice, Score)):
        raise TypeError("Expected a Predicate, Choice or Score question")
    _validate_text(question.question)
    if isinstance(question, Choice):
        _validate_options(question.choices, allow_booleans=True)
    elif isinstance(question, Score):
        _validate_options(question.levels, allow_booleans=False)


def validate_questions(questions: Mapping[str, Question]) -> None:
    if not isinstance(questions, Mapping):
        raise TypeError("Questions must be a mapping")
    if not questions:
        raise ValueError("Questions must not be empty")
    for name, question in questions.items():
        _validate_text(name)
        validate_question(question)
