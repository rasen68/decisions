"""Normalized answers and response metadata."""

from dataclasses import dataclass

from .inputs import JSONObject, JSONValue
from .questions import ChoiceValue

@dataclass
class PredicateAnswer:
    probability: float

@dataclass
class ChoiceAnswer:
    choice: ChoiceValue
    confidence: float
    probabilities: dict[ChoiceValue, float]

@dataclass
class ScoreAnswer:
    score: float
    confidence: float
    probabilities: dict[str, float]

Answer = PredicateAnswer | ChoiceAnswer | ScoreAnswer

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

@dataclass
class Result:
    answers: dict[str, Answer]
    provider: str
    model: str
    usage: Usage
    raw: JSONObject

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]
