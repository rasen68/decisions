from collections.abc import Mapping
from dataclasses import replace
from typing import assert_type

from decisions import (
    AsyncClient, Choice, ChoiceAnswer, Client, Image, Noul, Predicate,
    PredicateAnswer, RefusalAnswer, Result, Score, ScoreAnswer,
)
from decisions.providers import Jeff, OpenAI, Provider, TypeSafe
from decisions.types import Answer, JSONObject, JSONValue, Question, State


def sync_usage() -> None:
    provider = OpenAI(api_key="synthetic")
    assert_type(replace(provider, model="another-model"), OpenAI)
    labels = ["billing", "support"]
    descriptions = {"billing": "Payments", "support": "Technical help"}
    boolean_labels = [True, False]
    mixed_labels: list[str | bool] = [True, "unknown"]
    questions: dict[str, Question] = {
        "ready": Predicate("Ready?"),
        "team": Choice("Which team?", choices=labels),
        "urgency": Score("How urgent?", levels=labels),
    }
    typed_questions: Mapping[str, Question] = questions
    image = Image.from_bytes(b"synthetic", mime_type="image/png")
    with Client(provider) as client:
        assert_type(client, Client)
        assert_type(client.provider, Provider)
        answer = client.predicate("Ready?", "Text", images=[image])
        assert_type(answer, PredicateAnswer | RefusalAnswer)
        if isinstance(answer, RefusalAnswer):
            assert_type(answer.explanation, str | None)
        else:
            assert_type(answer, PredicateAnswer)
            assert_type(answer.probability, float)
        choice = client.choice("Which team?", choices=descriptions)
        assert_type(choice, ChoiceAnswer | RefusalAnswer)
        if isinstance(choice, ChoiceAnswer):
            assert_type(choice.choice, str | bool)
            assert_type(choice.probabilities, dict[str | bool, float])
        assert_type(client.choice("True?", choices=boolean_labels), ChoiceAnswer | RefusalAnswer)
        assert_type(client.choice("True?", choices={True: "Yes", False: "No"}),
                    ChoiceAnswer | RefusalAnswer)
        assert_type(client.choice("Which?", choices=mixed_labels),
                    ChoiceAnswer | RefusalAnswer)
        score = client.score("How urgent?", levels=labels)
        assert_type(score, ScoreAnswer | RefusalAnswer)
        if isinstance(score, ScoreAnswer):
            assert_type(score.score, float)
            assert_type(score.confidence, float)
            assert_type(score.probabilities, dict[str, float])
        assert_type(client.noul("Ready?"), PredicateAnswer | RefusalAnswer)
        result = client.decide(typed_questions, "Text")
        assert_type(result, Result)
        assert_type(result["ready"], Answer)
        assert_type(result.raw, JSONObject)
        assert_type(result.usage.input_tokens, int)
        assert_type(result.model, str)
        if isinstance(result["ready"], PredicateAnswer):
            assert_type(result["ready"].probability, float)
    state: State = {"text": "Refund", "flags": [True, None, 1]}
    raw_value: JSONValue = state
    assert_type(raw_value, JSONValue)
    with Client(TypeSafe(api_key="synthetic")) as client:
        assert_type(client.decide(questions, state), Result)
    with Client(Jeff("http://localhost:8765")) as client:
        assert_type(client.decide({"ready": Noul("Ready?")}, state), Result)


async def async_usage() -> None:
    async with AsyncClient(OpenAI(api_key="synthetic")) as client:
        assert_type(client, AsyncClient)
        assert_type(client.provider, Provider)
        answer = await client.predicate("Ready?")
        assert_type(answer, PredicateAnswer | RefusalAnswer)
        if not isinstance(answer, RefusalAnswer):
            assert_type(answer.probability, float)
        assert_type(await client.choice("Team?", choices=["billing", "support"]),
                    ChoiceAnswer | RefusalAnswer)
        assert_type(await client.score("Urgent?", levels=["low", "high"]),
                    ScoreAnswer | RefusalAnswer)
        assert_type(await client.noul("Ready?"), PredicateAnswer | RefusalAnswer)
        assert_type(await client.decide({"ready": Predicate("Ready?")}), Result)
    await client.aclose()
