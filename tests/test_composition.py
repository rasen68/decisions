"""Internal composition contracts, independent of provider registrations."""

from copy import deepcopy
import unittest

from decisions import Image, Predicate
from decisions.providers.composition import (
    Encoded, Header, HTTPBinding, Integration, PublicRequest, ResponsePipeline,
    Target, merge_headers,
)
from decisions.types import InvalidResponseError


class CompositionTests(unittest.TestCase):
    def test_headers_are_case_insensitive_and_auth_is_reserved(self):
        self.assertEqual(merge_headers((Header("X-Images", "yes"),),
                                      (Header("x-images", "yes"),)), {"X-Images": "yes"})
        for groups in (
            ((Header("X-Images", "yes"),), (Header("x-images", "no"),)),
            ((Header("authorization", "secret"),),),
            ((Header("Proxy-Authorization", "secret"),),),
        ):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                merge_headers(*groups)

    def test_binding_defaults_cannot_silently_override_requirements(self):
        target = Target("host", "model", "https://example.invalid", {"Accept": "application/json"})
        with self.assertRaises(ValueError):
            HTTPBinding()({}, (Header("accept", "text/plain"),), target)
        prepared = HTTPBinding()({}, (Header("accept", "application/json"),), target)
        self.assertEqual(prepared.headers, {"Accept": "application/json"})
        self.assertEqual(target.headers, {"Accept": "application/json"})

    def test_encoders_reject_incompatible_data_without_host_rules(self):
        from decisions import Choice
        from decisions.providers.requests import build_openai_input, build_systemone_question

        with self.assertRaises(ValueError):
            build_openai_input({"structured": "state"}, [Image(b"image", "image/png")])
        with self.assertRaises(ValueError):
            build_systemone_question(Choice("Which?", choices=[True, False]))

    def test_independent_components_wrappers_and_request_headers(self):
        state = {"text": ["original"]}
        image = Image.from_bytes(b"image", mime_type="image/png")
        request = PublicRequest({"check": Predicate("Check?")}, state, [image])
        original = deepcopy(request)
        events = []

        def rule(request, target):
            events.append("rule")
            if request.images and request.state is None:
                raise ValueError("images require state")

        def questions(request, target):
            events.append("questions")
            return Encoded(tuple(request.questions))

        def inputs(request, target):
            events.append("inputs")
            return Encoded({"state": request.state, "images": len(request.images)},
                           (Header("X-Images", "yes"),) if request.images else ())

        def assemble(questions, inputs, target):
            return {"wrapped": {"model": target.model, "questions": questions, "input": inputs}}

        def unwrap(raw):
            if not isinstance(raw, dict) or "payload" not in raw:
                raise InvalidResponseError
            return raw["payload"]

        def validate(request, payload):
            events.append("validate")
            if payload != "valid":
                raise InvalidResponseError
            return payload

        def decode(request, target, payload, raw):
            events.append("decode")
            return raw

        integration = Integration((rule,), questions, inputs, assemble, HTTPBinding(),
                                  ResponsePipeline(unwrap, validate, decode))
        target = Target("host", "model", "https://example.invalid", {"Authorization": "Bearer secret"})
        prepared = integration.prepare(request, target)
        self.assertEqual(events[:3], ["rule", "questions", "inputs"])
        self.assertEqual(prepared.method, "POST")
        self.assertEqual(prepared.body["wrapped"]["input"]["images"], 1)
        self.assertEqual(prepared.headers["X-Images"], "yes")
        self.assertNotIn("secret", repr(prepared))
        self.assertNotIn("secret", repr(target))
        plain = integration.prepare(PublicRequest(request.questions, "text", ()), target)
        self.assertNotIn("X-Images", plain.headers)
        self.assertEqual(request, original)
        raw = {"payload": "valid", "routing": {"upstream": "model"}}
        before = deepcopy(raw)
        self.assertIs(integration.response.decode(request, target, raw), raw)
        self.assertEqual(raw, before)
        self.assertEqual(events[-2:], ["validate", "decode"])
        with self.assertRaises(InvalidResponseError):
            integration.response.decode(request, target, {})
        events.clear()
        with self.assertRaises(InvalidResponseError):
            integration.response.decode(request, target, {"payload": "bad"})
        self.assertEqual(events, ["validate"])
        events.clear()
        with self.assertRaises(ValueError):
            integration.prepare(PublicRequest(request.questions, None, [image]), target)
        self.assertEqual(events, ["rule"])


if __name__ == "__main__":
    unittest.main()
