import unittest
from inspect import signature
from typing import get_args
from unittest.mock import AsyncMock, patch
import asyncio

import httpx

import decisions
from decisions import AsyncClient, Client, InvalidResponseError, Predicate
from decisions.types import Answer
import test_answer_validation as answer_tests
from test_response_structure import PROVIDERS


class RefusalTests(unittest.TestCase):
    def test_public_answer_type_and_single_question_annotations(self):
        refusal_type = getattr(decisions, "RefusalAnswer", None)
        self.assertIsNotNone(refusal_type)
        self.assertIn(refusal_type, get_args(Answer))
        from decisions.types import RefusalAnswer
        self.assertIs(refusal_type, RefusalAnswer)
        for client in (Client, AsyncClient):
            for method in ("predicate", "noul", "choice", "score"):
                self.assertIn(refusal_type, get_args(signature(getattr(client, method)).return_annotation))

    def test_single_question_refusals(self):
        helper = answer_tests.AnswerValidationTests()
        for kind in ("predicate", "choice", "score", "noul"):
            question, _ = answer_tests.question_and_answer(PROVIDERS[0], "predicate" if kind == "noul" else kind)
            for asynchronous in (False, True):
                with self.subTest(kind=kind, asynchronous=asynchronous):
                    result = helper.call(PROVIDERS[0], question, {"type": "refusal", "name": None}, asynchronous, method=kind)
                    self.assertIsInstance(result, decisions.RefusalAnswer)
                    self.assertIsNone(result.explanation)

    def test_mixed_answers_preserve_success_raw_and_order(self):
        raw = {"model": "reported", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": [
            {"type": "predicate", "probability": 0.8},
            {"type": "refusal", "name": "blocked", "extra": "future metadata"},
        ]}
        for asynchronous in (False, True):
            response = httpx.Response(200)
            questions = {"ready": Predicate("Ready?"), "blocked": Predicate("Blocked?")}
            with patch.object(response, "json", return_value=raw):
                if asynchronous:
                    async def run():
                        async with AsyncClient(PROVIDERS[0]) as client:
                            with patch.object(client, "_send", AsyncMock(return_value=response)):
                                return await client.decide(questions)
                    result = asyncio.run(run())
                else:
                    with Client(PROVIDERS[0]) as client:
                        with patch.object(client, "_send", return_value=response):
                            result = client.decide(questions)
            self.assertEqual(list(result.answers), ["ready", "blocked"])
            self.assertEqual(result["ready"].probability, 0.8)
            self.assertIsInstance(result["blocked"], decisions.RefusalAnswer)
            self.assertIs(result.raw, raw)

    def test_malformed_and_unsupported_refusals_are_rejected(self):
        helper = answer_tests.AnswerValidationTests()
        question = Predicate("Ready?")
        for provider in PROVIDERS:
            cases = [{"type": "refusal"}, {"type": "refusal", "name": 5},
                     {"type": "refusal", "name": []}, {"type": "future", "name": None}]
            if provider is not PROVIDERS[0]:
                cases.append({"type": "refusal", "name": None})
            for answer in cases:
                helper.reject(provider, question, answer)


if __name__ == "__main__":
    unittest.main()
