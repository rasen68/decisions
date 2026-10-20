import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from decisions import AsyncClient, Client, Image, Predicate
from decisions.providers import OpenAI
from decisions.providers.requests import build_request
from test_response_structure import PROVIDERS


METHODS = ("decide", "predicate", "noul", "choice", "score")


def arguments(method):
    if method == "decide":
        return {"answer": Predicate("Ready?")}, {}
    if method == "choice":
        return "Which?", {"choices": ["a", "b"]}
    if method == "score":
        return "How much?", {"levels": ["low", "high"]}
    return "Ready?", {}


def response_for(provider, method):
    openai = isinstance(provider, OpenAI)
    if method == "choice":
        answer = {"type": "choice", "choice": "a", "confidence": 0.8,
                  "probabilities": [{"value": "a", "probability": 0.8}, {"value": "b", "probability": 0.2}] if openai else {"a": 0.8, "b": 0.2}}
    elif method == "score":
        answer = {"type": "score", "score": 0.2, "confidence": 0.8,
                  "probabilities": [{"value": 0, "label": "low", "probability": 0.8}, {"value": 1, "label": "high", "probability": 0.2}] if openai else {"0": 0.8, "1": 0.2}}
    else:
        answer = {"type": "predicate" if openai else "noul", "probability" if openai else "noul": 0.8}
    return httpx.Response(200, json={"model": "reported", "usage": {"input_tokens": 1, "output_tokens": 1},
                                    "answers": [answer] if openai else {"answer": answer}})


class StateRequestTests(unittest.TestCase):
    def call(self, provider, state, method, asynchronous, expected=None):
        argument, kwargs = arguments(method)
        response = response_for(provider, method)
        if asynchronous:
            async def run():
                async with AsyncClient(provider) as client:
                    with patch.object(client, "_send", AsyncMock(return_value=response)) as send:
                        if expected:
                            with self.assertRaises(expected):
                                await getattr(client, method)(argument, state=state, **kwargs)
                            send.assert_not_called()
                        else:
                            await getattr(client, method)(argument, state=state, **kwargs)
                            send.assert_awaited_once()
                            return send.call_args.args[0]
            return asyncio.run(run())
        with Client(provider) as client:
            with patch.object(client, "_send", return_value=response) as send:
                if expected:
                    with self.assertRaises(expected):
                        getattr(client, method)(argument, state=state, **kwargs)
                    send.assert_not_called()
                else:
                    getattr(client, method)(argument, state=state, **kwargs)
                    send.assert_called_once()
                    return send.call_args.args[0]

    def assert_rejected(self, state, expected, providers=PROVIDERS):
        for provider in providers:
            for method in METHODS:
                for asynchronous in (False, True):
                    with self.subTest(provider=provider.name, method=method, asynchronous=asynchronous):
                        self.call(provider, state, method, asynchronous, expected)

    def test_invalid_state_types_cause_no_http_call(self):
        for state in (True, 1, 1.5, b"text", ("text",), {"text"}, {"outer": [b"text"]}, [{1: "value"}]):
            self.assert_rejected(state, TypeError)

    def test_nonfinite_and_cyclic_state_cause_no_http_call(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.assert_rejected({"outer": [value]}, ValueError)
        cycle = []
        cycle.append({"back": cycle})
        self.assert_rejected(cycle, ValueError)

    def test_openai_rejects_structured_state(self):
        for state in ({}, [], {"nested": [True, None, 0.5]}, ["text"]):
            self.assert_rejected(state, ValueError, PROVIDERS[:1])

    def test_openai_rejects_structured_state_with_images(self):
        for state in ({}, []):
            with self.assertRaises(ValueError):
                build_request(PROVIDERS[0], {"answer": Predicate("Ready?")}, state, [Image(b"image", "image/png")])

    def test_text_and_omission_remain_supported_by_every_method(self):
        for provider in PROVIDERS:
            for state in (None, "", " \n private-state \t"):
                for method in METHODS:
                    for asynchronous in (False, True):
                        with self.subTest(provider=provider.name, state=state, method=method, asynchronous=asynchronous):
                            body = self.call(provider, state, method, asynchronous)
                            if isinstance(provider, OpenAI):
                                self.assertIs(body["input"], state)
                            elif state is None:
                                self.assertEqual(body["state"], "")
                            else:
                                self.assertIs(body["state"], state)

    def test_systemone_preserves_nested_and_shared_state(self):
        shared = [None, True, False, -2, 0.5, "text", {"": []}]
        for provider in PROVIDERS[1:]:
            for state in ({}, [], {"left": shared, "right": shared}, [shared, shared]):
                before = json.dumps(state, allow_nan=False)
                for method in METHODS:
                    for asynchronous in (False, True):
                        with self.subTest(provider=provider.name, method=method, asynchronous=asynchronous):
                            body = self.call(provider, state, method, asynchronous)
                            self.assertIs(body["state"], state)
                            self.assertEqual(json.dumps(state, allow_nan=False), before)
                            self.assertEqual(json.loads(json.dumps(body))["state"], json.loads(before))

    def test_nested_state_is_revalidated_after_mutation(self):
        nested = ["valid"]
        state = {"outer": nested}
        build_request(PROVIDERS[1], {"answer": Predicate("Ready?")}, state, ())
        nested.append(b"invalid")
        self.assert_rejected(state, TypeError)


if __name__ == "__main__":
    unittest.main()
