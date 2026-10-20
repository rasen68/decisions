import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, patch

import httpx

import decisions
from decisions import AsyncClient, Choice, Client, InvalidResponseError, Predicate, Score
from decisions.providers import OpenAI
from test_response_structure import PROVIDERS


BAD_NUMBERS = (None, True, False, "0.5", [], {}, float("nan"), float("inf"), -float("inf"), -0.01, 1.01, 10**400)


def question_and_answer(provider, kind):
    openai = isinstance(provider, OpenAI)
    if kind == "predicate":
        return Predicate("Ready?"), {"type": "predicate" if openai else "noul", "probability" if openai else "noul": 0.5}
    if kind == "choice":
        return Choice("Which?", choices=["a", "b"]), {
            "type": "choice", "choice": "a", "confidence": 0.7,
            "probabilities": [{"value": "b", "probability": 0.2}, {"value": "a", "probability": 0.7}] if openai else {"b": 0.2, "a": 0.7},
        }
    return Score("How much?", levels=["low", "high"]), {
        "type": "score", "score": 0.4, "confidence": 0.7,
        "probabilities": [{"value": 1, "label": "high", "probability": 0.2}, {"value": 0, "label": "low", "probability": 0.7}] if openai else {"1": 0.2, "0": 0.7},
    }


class AnswerValidationTests(unittest.TestCase):
    def call(self, provider, question, answer, asynchronous, method="decide"):
        raw = {"model": "reported", "usage": {"input_tokens": 0, "output_tokens": 1},
               "answers": [answer] if isinstance(provider, OpenAI) else {"answer": answer}}
        response = httpx.Response(200)
        args = ({"answer": question},) if method == "decide" else (question.question,)
        kwargs = {}
        if method == "choice":
            kwargs["choices"] = question.choices
        elif method == "score":
            kwargs["levels"] = question.levels
        with patch.object(response, "json", return_value=raw):
            if asynchronous:
                async def run():
                    async with AsyncClient(provider) as client:
                        with patch.object(client, "_send", AsyncMock(return_value=response)) as send:
                            try:
                                return await getattr(client, method)(*args, **kwargs)
                            finally:
                                send.assert_awaited_once()
                return asyncio.run(run())
            with Client(provider) as client:
                with patch.object(client, "_send", return_value=response) as send:
                    try:
                        return getattr(client, method)(*args, **kwargs)
                    finally:
                        send.assert_called_once()

    def reject(self, provider, question, answer):
        for asynchronous in (False, True):
            with self.subTest(provider=provider.name, asynchronous=asynchronous, answer=answer):
                with self.assertRaises(InvalidResponseError) as caught:
                    self.call(provider, question, answer, asynchronous)
                self.assertEqual(str(caught.exception), "")

    def test_predicate_numeric_values(self):
        for provider in PROVIDERS:
            question, answer = question_and_answer(provider, "predicate")
            field = "probability" if isinstance(provider, OpenAI) else "noul"
            for value in BAD_NUMBERS:
                self.reject(provider, question, {**answer, field: value})

    def test_valid_values_are_preserved_without_normalization(self):
        for provider in PROVIDERS:
            for kind in ("predicate", "choice", "score"):
                question, answer = question_and_answer(provider, kind)
                for asynchronous in (False, True):
                    with self.subTest(provider=provider.name, kind=kind, asynchronous=asynchronous):
                        result = self.call(provider, question, answer, asynchronous)
                        self.assertIs(result.raw["answers"][0 if isinstance(provider, OpenAI) else "answer"], answer)
                        if kind != "predicate":
                            probabilities = result["answer"].probabilities
                            self.assertEqual(list(probabilities), ["a", "b"] if kind == "choice" else ["low", "high"])
                            self.assertEqual(list(probabilities.values()), [0.7, 0.2])
                field = "probability" if isinstance(provider, OpenAI) else "noul"
                for value in (0, 1, 0.0, 1.0):
                    boundary = dict(answer)
                    boundary[field if kind == "predicate" else "confidence"] = value
                    if kind == "score":
                        boundary["score"] = value
                    for asynchronous in (False, True):
                        result = self.call(provider, question, boundary, asynchronous)["answer"]
                        self.assertIs(getattr(result, "probability" if kind == "predicate" else "confidence"), value)

    def test_answer_kind_must_match(self):
        for provider in PROVIDERS:
            for kind in ("predicate", "choice", "score"):
                question, answer = question_and_answer(provider, kind)
                for value in (None, [], "future", "score" if kind != "score" else "choice"):
                    self.reject(provider, question, {**answer, "type": value})
                answer.pop("type")
                self.reject(provider, question, answer)

    def test_choice_and_score_required_fields(self):
        for provider in PROVIDERS:
            for kind in ("choice", "score"):
                question, answer = question_and_answer(provider, kind)
                for field in (kind, "confidence", "probabilities"):
                    malformed = dict(answer)
                    del malformed[field]
                    self.reject(provider, question, malformed)

    def test_confidence_score_and_distribution_values(self):
        for provider in PROVIDERS:
            for kind in ("choice", "score"):
                question, answer = question_and_answer(provider, kind)
                for value in BAD_NUMBERS:
                    self.reject(provider, question, {**answer, "confidence": value})
                    if kind == "score":
                        self.reject(provider, question, {**answer, "score": value})
                    malformed = copy.deepcopy(answer)
                    probabilities = malformed["probabilities"]
                    if isinstance(provider, OpenAI):
                        probabilities[0]["probability"] = value
                    else:
                        probabilities[next(iter(probabilities))] = value
                    self.reject(provider, question, malformed)

    def test_distribution_shape_coverage_and_duplicates(self):
        for provider in PROVIDERS:
            for kind in ("choice", "score"):
                question, answer = question_and_answer(provider, kind)
                cases = [None, True, "invalid", 1, [], {}]
                probabilities = answer["probabilities"]
                if isinstance(provider, OpenAI):
                    cases += [probabilities[:1], probabilities + [probabilities[0]], [None, probabilities[1]], [{}, probabilities[1]]]
                    for field in ("value", "probability") if kind == "choice" else ("label", "probability", "value"):
                        malformed = copy.deepcopy(probabilities)
                        del malformed[0][field]
                        cases.append(malformed)
                    for value in (None, [], {}, True, 1.0, "unknown"):
                        malformed = copy.deepcopy(probabilities)
                        malformed[0]["value" if kind == "choice" else "label"] = value
                        cases.append(malformed)
                    if kind == "score":
                        for value in (True, 1.0, -1, 2, "1", None, 0):
                            malformed = copy.deepcopy(probabilities)
                            malformed[0]["value"] = value
                            cases.append(malformed)
                else:
                    cases += [{next(iter(probabilities)): 0.5}, {**probabilities, "unknown": 0.1}]
                for distribution in cases:
                    self.reject(provider, question, {**answer, "probabilities": distribution})

    def test_score_bounds_follow_requested_level_count(self):
        for provider in PROVIDERS:
            question, answer = question_and_answer(provider, "score")
            question.levels = ["low", "middle", "high"]
            answer["probabilities"] = (
                [{"value": index, "label": label, "probability": 0.33} for index, label in enumerate(question.levels)]
                if isinstance(provider, OpenAI) else {"0": 0.33, "1": 0.33, "2": 0.33}
            )
            for value in (0, 1.75, 2):
                for asynchronous in (False, True):
                    result = self.call(provider, question, {**answer, "score": value}, asynchronous)
                    self.assertIs(result["answer"].score, value)
            self.reject(provider, question, {**answer, "score": 2.01})

    def test_invalid_later_answer_prevents_any_conversion(self):
        from decisions.providers.responses import decode_result
        from decisions import Predicate
        for provider in PROVIDERS:
            _, valid = question_and_answer(provider, "predicate")
            raw = {"model": "reported", "usage": {"input_tokens": 0, "output_tokens": 1},
                   "answers": [valid, {}] if isinstance(provider, OpenAI) else {"first": valid, "last": {}}}
            with patch("decisions.providers.responses.decode_answer") as decode:
                with self.assertRaises(InvalidResponseError):
                    decode_result(provider, {"first": Predicate("Ready?"), "last": Predicate("Safe?")}, raw)
                decode.assert_not_called()

    def test_choice_selection_requires_matching_public_type_and_value(self):
        for provider in PROVIDERS:
            question, answer = question_and_answer(provider, "choice")
            for value in (None, [], {}, True, 1, "unknown"):
                self.reject(provider, question, {**answer, "choice": value})
        provider = PROVIDERS[0]
        question = Choice("Which?", choices=[True, "true"])
        answer = {"type": "choice", "choice": True, "confidence": 1,
                  "probabilities": [{"value": "true", "probability": 0}, {"value": True, "probability": 1}]}
        for asynchronous in (False, True):
            result = self.call(provider, question, answer, asynchronous)["answer"]
            self.assertIs(result.choice, True)
            self.assertEqual(list(result.probabilities), [True, "true"])
        self.reject(provider, question, {**answer, "choice": 1})
        malformed = copy.deepcopy(answer)
        malformed["probabilities"][1]["value"] = 1
        self.reject(provider, question, malformed)


if __name__ == "__main__":
    unittest.main()
