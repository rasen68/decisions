"""Reusable provider-independent question definitions."""

from collections.abc import Mapping
from dataclasses import dataclass, field

ChoiceValue = str | bool
Choices = list[ChoiceValue] | Mapping[ChoiceValue, str]
Levels = list[str] | Mapping[str, str]

@dataclass
class Predicate:
    question: str

Noul = Predicate

@dataclass
class Choice:
    question: str
    choices: Choices = field(kw_only=True)

@dataclass
class Score:
    question: str
    levels: Levels = field(kw_only=True)

Question = Predicate | Choice | Score
