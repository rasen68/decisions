"""Deliberately different internal combinations, not advertised provider support."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import unittest
from unittest.mock import patch

import httpx

from decisions import AsyncClient, Client, Image, Predicate
from decisions.providers import OpenAI
from decisions.providers.composition import (
    Encoded, Header, HTTPBinding, Integration, PublicRequest, ResponsePipeline, Target,
)
from decisions.providers.integrations import (
    OPENAI, decode_systemone, identity_payload, separate_image_input,
    systemone_body, systemone_questions, validate_systemone,
)
from decisions.providers.requests import data_url
from decisions.types import HTTPError, InvalidResponseError


@dataclass(frozen=True)
class ReferencedInput:
    state: dict
    assets: dict


class IntegrationCompositionTests(unittest.TestCase):
    def test_systemone_questions_with_references_or_separate_images(self):
        def references(request, target):
            # Synthetic format with a deliberately reserved asset location.
            state = {"document": deepcopy(request.state),
                     "attachments": [f"asset:{index}" for index in range(len(request.images))]}
            return Encoded(ReferencedInput(state, {
                f"asset:{index}": data_url(image) for index, image in enumerate(request.images)
            }))

        def referenced_body(questions, inputs, target):
            return {"questions": questions.items, "state": inputs.state, "assets": inputs.assets}

        pipeline = ResponsePipeline(identity_payload, validate_systemone, decode_systemone)
        referenced = Integration((), systemone_questions, references, referenced_body, HTTPBinding(), pipeline)
        separate = Integration((), systemone_questions, separate_image_input, systemone_body, HTTPBinding(), pipeline)
        images = [Image(b"first", "image/png"), Image(b"second", "image/jpeg")]
        state = {"text": ["asset:0", "original"]}
        request = PublicRequest({"check": Predicate("Check?")}, state, images)
        before = deepcopy(request)
        target = Target("synthetic-host", "model", "https://example.invalid")
        ref_body = referenced.prepare(request, target).body
        image_body = separate.prepare(request, target).body
        self.assertEqual(ref_body["questions"], image_body["questions"])
        self.assertEqual(ref_body["questions"]["check"]["type"], "noul")
        self.assertEqual(ref_body["state"]["document"], state)
        self.assertIsNot(ref_body["state"]["document"], state)
        self.assertEqual(ref_body["state"]["attachments"], ["asset:0", "asset:1"])
        self.assertEqual(list(ref_body["assets"].values()), image_body["images"])
        self.assertIs(image_body["state"], state)
        self.assertEqual(request, before)

    def wrapped_integration(self):
        def inputs(request, target):
            encoded = OPENAI.inputs(request, target)
            return Encoded(encoded.value, (Header("X-Image-Opt-In", "true"),) if request.images else ())

        def body(questions, inputs, target):
            return {"request": OPENAI.assemble(questions, inputs, target), "route": "upstream"}

        def unwrap(raw):
            if not isinstance(raw, dict) or not isinstance(raw.get("response"), dict):
                raise InvalidResponseError
            return raw["response"]

        return replace(OPENAI, inputs=inputs, assemble=body, bind=HTTPBinding("PUT"),
                       response=replace(OPENAI.response, unwrap=unwrap))

    def test_both_clients_execute_identical_prepared_requests_without_header_leaks(self):
        integration = self.wrapped_integration()
        provider = OpenAI(api_key="synthetic-secret", model="selected-model")
        image = Image(b"image", "image/png")
        questions = {"check": Predicate("Check?")}
        raw = {"response": {"model": "upstream-model", "answers": [{"type": "predicate", "probability": 0.7}],
                            "usage": {"input_tokens": 2, "output_tokens": 1}},
               "routing": {"host": "synthetic-upstream"}}
        before = deepcopy((questions, image, raw))
        captured = []
        prepared = []

        def handler(request):
            captured.append((request.method, str(request.url), dict(request.headers), json.loads(request.content)))
            return httpx.Response(200, json=raw)

        def run_sync():
            with Client(provider) as client:
                client._http.close()
                client._http = httpx.Client(transport=httpx.MockTransport(handler))
                self.assertEqual(client.decide(questions, "state", images=[image]).raw, raw)
                result = client.decide(questions, "state")
                self.assertEqual(result.provider, "openai")
                self.assertEqual(result.model, "upstream-model")
                self.assertEqual(result["check"].probability, 0.7)
                self.assertEqual(result.raw, raw)

        async def run_async():
            async with AsyncClient(provider) as client:
                await client._http.aclose()
                client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                self.assertEqual((await client.decide(questions, "state", images=[image])).raw, raw)
                self.assertEqual((await client.decide(questions, "state")).raw, raw)

        original_prepare = Integration.prepare
        def capture_prepare(self, request, target):
            value = original_prepare(self, request, target)
            prepared.append(value)
            return value

        with patch("decisions.client.resolve", return_value=integration), patch.object(Integration, "prepare", capture_prepare):
            run_sync()
            asyncio.run(run_async())
        self.assertEqual(prepared[:2], prepared[2:])
        self.assertEqual(captured[:2], captured[2:])
        self.assertEqual(captured[0][0], "PUT")
        self.assertIn("x-image-opt-in", captured[0][2])
        self.assertNotIn("x-image-opt-in", captured[1][2])
        self.assertEqual(captured[0][2]["authorization"], "Bearer synthetic-secret")
        self.assertEqual(captured[0][3]["request"]["questions"][0]["type"], "predicate")
        self.assertEqual(captured[0][3]["request"]["input"][0]["content"][1]["type"], "input_image")
        self.assertEqual((questions, image, raw), before)

    def test_conditional_rules_and_header_conflicts_fail_before_http(self):
        def require_state(request, target):
            if request.images and request.state is None:
                raise ValueError("images require state")

        def questions(request, target):
            return replace(OPENAI.questions(request, target), headers=(Header("X-Format", "one"),))

        def inputs(request, target):
            return replace(OPENAI.inputs(request, target), headers=(Header("x-format", "two"),))

        integrations = (
            (replace(OPENAI, rules=(require_state,)), None, [Image(b"image", "image/png")]),
            (replace(OPENAI, questions=questions, inputs=inputs), "text", []),
        )
        for integration, state, images in integrations:
            for asynchronous in (False, True):
                with self.subTest(asynchronous=asynchronous, state=state):
                    def handler(request):
                        self.fail("Invalid input reached HTTP")
                    async def run():
                        async with AsyncClient(OpenAI(api_key="synthetic")) as client:
                            await client._http.aclose()
                            client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                            with self.assertRaises(ValueError):
                                await client.predicate("Check?", state, images=images)
                    with patch("decisions.client.resolve", return_value=integration):
                        if asynchronous:
                            asyncio.run(run())
                        else:
                            with Client(OpenAI(api_key="synthetic")) as client:
                                client._http.close()
                                client._http = httpx.Client(transport=httpx.MockTransport(handler))
                                with self.assertRaises(ValueError):
                                    client.predicate("Check?", state, images=images)

    def test_malformed_envelopes_and_http_precedence(self):
        integration = self.wrapped_integration()
        request = PublicRequest({"check": Predicate("Check?")}, "state", ())
        target = Target("host", "model", "https://example.invalid")
        raw = {"response": {"model": "upstream", "answers": [{"type": "predicate", "probability": 0.6}],
                            "usage": {"input_tokens": 1, "output_tokens": 1}}, "route": ["host", "upstream"]}
        before = deepcopy(raw)
        result = integration.response.decode(request, target, raw)
        self.assertIs(result.raw, raw)
        self.assertEqual(result.provider, "host")
        self.assertEqual(raw, before)
        for raw in (None, [], {}, {"response": []}, {"response": {}},
                    {"response": {"model": "model", "answers": [], "usage": {}}}):
            with self.subTest(raw=raw), self.assertRaises(InvalidResponseError):
                integration.response.decode(request, target, raw)
        with patch("decisions.client.resolve", return_value=integration):
            with Client(OpenAI(api_key="synthetic")) as client:
                with patch.object(client, "_send", return_value=httpx.Response(503, json={})):
                    with patch.object(ResponsePipeline, "decode") as decode:
                        with self.assertRaises(HTTPError):
                            client.decide(request.questions)
                        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
