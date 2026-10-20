import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from decisions import AsyncClient, Client, InvalidResponseError, Predicate
from decisions.providers import Jeff, OpenAI, TypeSafe


PROVIDERS = (
    OpenAI(api_key="synthetic-secret"),
    TypeSafe(api_key="synthetic-secret"),
    Jeff("http://localhost:8765", api_key="synthetic-secret"),
)
QUESTIONS = {"first": Predicate("Ready?"), "second": Predicate("Safe?")}


def valid_response(provider):
    if isinstance(provider, OpenAI):
        answers = [{"type": "predicate", "probability": 0.8}, {"type": "predicate", "probability": 0.2}]
    else:
        answers = {"second": {"type": "noul", "noul": 0.2}, "first": {"type": "noul", "noul": 0.8}}
    return {
        "answers": answers,
        "model": "reported-model",
        "usage": {"input_tokens": 0, "output_tokens": 3},
    }


class ResponseStructureTests(unittest.TestCase):
    def call_client(self, provider, raw, asynchronous):
        response = httpx.Response(200)
        with patch.object(response, "json", return_value=raw):
            if asynchronous:
                async def call():
                    async with AsyncClient(provider) as client:
                        with patch.object(client, "_send", AsyncMock(return_value=response)) as send:
                            try:
                                return await client.decide(QUESTIONS, "private-state")
                            finally:
                                send.assert_awaited_once()
                return asyncio.run(call())
            with Client(provider) as client:
                with patch.object(client, "_send", return_value=response) as send:
                    try:
                        return client.decide(QUESTIONS, "private-state")
                    finally:
                        send.assert_called_once()

    def assert_rejected(self, make_raw):
        for provider in PROVIDERS:
            for asynchronous in (False, True):
                with self.subTest(provider=provider.name, asynchronous=asynchronous):
                    raw = make_raw(provider)
                    before = copy.deepcopy(raw)
                    with self.assertRaises(InvalidResponseError) as caught:
                        self.call_client(provider, raw, asynchronous)
                    for private in ("synthetic-secret", "private-state"):
                        self.assertNotIn(private, str(caught.exception))
                    self.assertEqual(raw, before)

    def test_response_must_be_an_object(self):
        for raw in (None, [], "private-state synthetic-secret", 7, True):
            with self.subTest(raw=raw):
                self.assert_rejected(lambda provider: raw)

    def test_answers_container_is_required(self):
        def missing(provider):
            raw = valid_response(provider)
            del raw["answers"]
            return raw
        self.assert_rejected(missing)

    def test_answers_container_has_provider_specific_type(self):
        for value in (None, "private-state", 7, True):
            def malformed(provider):
                raw = valid_response(provider)
                raw["answers"] = value
                return raw
            with self.subTest(value=value):
                self.assert_rejected(malformed)
        def wrong_container(provider):
            raw = valid_response(provider)
            if isinstance(provider, OpenAI):
                raw["answers"] = {"first": {"probability": 0.8}, "second": {"probability": 0.2}}
            else:
                raw["answers"] = [{"noul": 0.8}, {"noul": 0.2}]
            return raw
        self.assert_rejected(wrong_container)

    def test_every_requested_answer_is_required(self):
        for missing_all in (False, True):
            def incomplete(provider):
                raw = valid_response(provider)
                if missing_all:
                    raw["answers"] = [] if isinstance(provider, OpenAI) else {}
                elif isinstance(provider, OpenAI):
                    raw["answers"].pop()
                else:
                    del raw["answers"]["second"]
                return raw
            with self.subTest(missing_all=missing_all):
                self.assert_rejected(incomplete)

    def test_extra_positional_answers_are_rejected(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                provider = PROVIDERS[0]
                raw = valid_response(provider)
                raw["answers"].append({"probability": 0.5})
                with self.assertRaises(InvalidResponseError):
                    self.call_client(provider, raw, asynchronous)

    def test_each_requested_answer_must_be_an_object(self):
        for value in (None, [], "private-state synthetic-secret", 0, True):
            for index, name in enumerate(QUESTIONS):
                def malformed(provider):
                    raw = valid_response(provider)
                    key = index if isinstance(provider, OpenAI) else name
                    raw["answers"][key] = value
                    return raw
                with self.subTest(value=value, name=name):
                    self.assert_rejected(malformed)

    def test_model_is_a_required_string(self):
        def missing(provider):
            raw = valid_response(provider)
            del raw["model"]
            return raw
        self.assert_rejected(missing)
        for value in (None, 7, True, [], {"private-state": "synthetic-secret"}):
            def malformed(provider):
                raw = valid_response(provider)
                raw["model"] = value
                return raw
            with self.subTest(value=value):
                self.assert_rejected(malformed)

    def test_usage_is_a_required_object(self):
        def missing(provider):
            raw = valid_response(provider)
            del raw["usage"]
            return raw
        self.assert_rejected(missing)
        for value in (None, [], "private-state", 7, True):
            def malformed(provider):
                raw = valid_response(provider)
                raw["usage"] = value
                return raw
            with self.subTest(value=value):
                self.assert_rejected(malformed)

    def test_token_counts_are_required_nonnegative_integers(self):
        for field in ("input_tokens", "output_tokens"):
            def missing(provider):
                raw = valid_response(provider)
                del raw["usage"][field]
                return raw
            with self.subTest(field=field, missing=True):
                self.assert_rejected(missing)
            for value in (None, -1, 1.0, 0.5, True, False, "private-state", [], {}):
                def malformed(provider):
                    raw = valid_response(provider)
                    raw["usage"][field] = value
                    return raw
                with self.subTest(field=field, value=value):
                    self.assert_rejected(malformed)

    def test_valid_response_preserves_order_raw_and_extra_metadata(self):
        for provider in PROVIDERS:
            for asynchronous in (False, True):
                with self.subTest(provider=provider.name, asynchronous=asynchronous):
                    raw = valid_response(provider)
                    raw["extra"] = {"private-state": [1, 2]}
                    raw["usage"]["extra"] = "future metadata"
                    if isinstance(provider, OpenAI):
                        raw["answers"][0]["extra"] = "future metadata"
                    else:
                        raw["answers"]["first"]["extra"] = "future metadata"
                        raw["answers"]["unrequested"] = None
                    before = copy.deepcopy(raw)
                    result = self.call_client(provider, raw, asynchronous)
                    self.assertEqual(list(result.answers), list(QUESTIONS))
                    self.assertEqual(result["first"].probability, 0.8)
                    self.assertEqual(result["second"].probability, 0.2)
                    self.assertEqual(result.model, "reported-model")
                    self.assertEqual(result.provider, provider.name)
                    self.assertEqual(result.usage.input_tokens, 0)
                    self.assertEqual(result.usage.output_tokens, 3)
                    self.assertIs(result.raw, raw)
                    self.assertEqual(raw, before)


if __name__ == "__main__":
    unittest.main()
