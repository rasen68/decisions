import tempfile
from pathlib import Path
import unittest

from decisions import Choice, Image, Noul, Predicate, Score
from decisions.providers.conversion import build_request, decode_result
from decisions.providers import Jeff, OpenAI, TypeSafe


class RepresentationTests(unittest.TestCase):
    def test_json_aliases_and_raw_annotation(self):
        from typing import get_type_hints
        from decisions.types import JSONObject, JSONValue, Result, Usage

        self.assertEqual(JSONObject, dict[str, JSONValue])
        self.assertIn("raw", get_type_hints(Result))
        raw: JSONObject = {"nested": {"items": [True, None, 1, 1.5, "text"]}}
        result = Result({}, "test", "test", Usage(0, 0), raw)
        self.assertIs(result.raw, raw)

    def test_questions_and_image_helpers(self):
        self.assertIs(Noul, Predicate)
        self.assertEqual(Noul("Ready?"), Predicate("Ready?"))
        self.assertEqual(Choice("Team?", choices=[True, "true"]).choices, [True, "true"])
        self.assertEqual(Score("Urgency?", levels={"low": "Later"}).levels, {"low": "Later"})
        self.assertEqual(Image.from_bytes(b"photo", mime_type="image/png"), Image(b"photo", "image/png"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "photo.jpg"
            path.write_bytes(b"photo")
            self.assertEqual(Image.from_path(path), Image(b"photo", "image/jpeg"))
            self.assertEqual(Image.from_path(str(path), mime_type="image/webp"), Image(b"photo", "image/webp"))

    def test_openai_request_preserves_typed_options(self):
        questions = {"yes": Predicate("Ready?"), "team": Choice("Team?", choices=[True, "true"]), "urgency": Score("Urgent?", levels={"low": "Later", "high": "Now"})}
        body = build_request(OpenAI(), questions, "Text", ())
        self.assertEqual(body, {"model": "gpt-6-luna", "input": "Text", "questions": [
            {"name": "yes", "type": "predicate", "instructions": "Ready?"},
            {"name": "team", "type": "choice", "instructions": "Team?", "choices": [{"value": True, "description": "True"}, {"value": "true", "description": "true"}]},
            {"name": "urgency", "type": "score", "instructions": "Urgent?", "levels": [{"label": "low", "description": "Later"}, {"label": "high", "description": "Now"}]},
        ]})

    def test_images_openai_and_jeff(self):
        images = [Image(b"a", "image/png"), Image(b"b", "image/jpeg")]
        questions = {"yes": Predicate("Ready?")}
        parts = [{"type": "input_image", "image_url": "data:image/png;base64,YQ=="}, {"type": "input_image", "image_url": "data:image/jpeg;base64,Yg=="}]
        self.assertEqual(build_request(OpenAI(), questions, None, images)["input"], [{"role": "user", "content": parts}])
        self.assertEqual(build_request(OpenAI(), questions, "Text", images)["input"][0]["content"], [{"type": "input_text", "text": "Text"}] + parts)
        body = build_request(Jeff("http://localhost:8765"), questions, None, images)
        self.assertEqual(body["state"], "")
        self.assertEqual(body["images"], [part["image_url"] for part in parts])

    def test_systemone_state_and_choice_roundtrip(self):
        questions = {"yes": Predicate("Ready?"), "team": Choice("Team?", choices={"billing": "Billing", "sales": "Sales", "option_0": "Literal", "support": "Support"}), "urgency": Score("Urgent?", levels=["low", "high"])}
        state = {"fixed": {"items": [1, None, True]}, "changing": "Text"}
        for provider in (TypeSafe(), Jeff("http://localhost:8765")):
            with self.subTest(provider=provider):
                body = build_request(provider, questions, state, ())
                self.assertIs(body["state"], state)
                self.assertEqual(list(body["questions"]), list(questions))
                self.assertEqual(body["questions"]["yes"]["type"], "noul")
                criteria = body["questions"]["team"]["criteria"]
                keys = list(criteria)
                self.assertEqual(criteria, {"billing": "Billing", "sales": "Sales", "option_0": "Literal", "support": "Support"})
                self.assertEqual(keys, ["billing", "sales", "option_0", "support"])
                self.assertEqual(body["questions"]["urgency"]["criteria"], ["low", "high"])
                raw = {"model": "reported-model", "answers": {
                    "urgency": {"type": "score", "score": .75, "confidence": .123, "probabilities": {"1": .75, "0": .25}},
                    "team": {"type": "choice", "choice": keys[0], "confidence": .678, "probabilities": dict(zip(reversed(keys), [.1, .2, .3, .4]))},
                    "yes": {"type": "noul", "noul": .9}}, "usage": {"input_tokens": 12, "output_tokens": 3, "orders": 1}, "extra": True}
                result = decode_result(provider, questions, raw)
                self.assertEqual(list(result.answers), list(questions))
                self.assertEqual(result["yes"].probability, .9)
                self.assertEqual(result["team"].choice, "billing")
                self.assertEqual(list(result["team"].probabilities), ["billing", "sales", "option_0", "support"])
                self.assertEqual(result["team"].probabilities["billing"], .4)
                self.assertEqual(result["team"].confidence, .678)
                self.assertEqual(result["urgency"].score, .75)
                self.assertEqual(result["urgency"].confidence, .123)
                self.assertEqual(result["urgency"].probabilities, {"low": .25, "high": .75})
                self.assertEqual(result.model, "reported-model")
                self.assertEqual(result.provider, "typesafe" if isinstance(provider, TypeSafe) else "jeff")
                self.assertEqual((result.usage.input_tokens, result.usage.output_tokens), (12, 3))
                self.assertIs(result.raw, raw)

    def test_openai_decode_order_and_public_labels(self):
        questions = {"team": Choice("Team?", choices=[False, "false"]), "urgency": Score("Urgent?", levels=["low", "high"]), "yes": Predicate("Ready?")}
        raw = {"model": "reported", "answers": [
            {"type": "choice", "choice": False, "confidence": .123, "probabilities": [{"value": "false", "probability": .2}, {"value": False, "probability": .8}]},
            {"type": "score", "score": .8, "confidence": .456, "probabilities": [{"value": 1, "label": "high", "probability": .8}, {"value": 0, "label": "low", "probability": .2}]},
            {"type": "predicate", "probability": .9}], "usage": {"input_tokens": 10, "output_tokens": 5}}
        result = decode_result(OpenAI(), questions, raw)
        self.assertIs(result["team"].choice, False)
        self.assertEqual(list(result["team"].probabilities.items()), [(False, .8), ("false", .2)])
        self.assertEqual(list(result["urgency"].probabilities.items()), [("low", .2), ("high", .8)])
        self.assertEqual(result["yes"].probability, .9)
        self.assertEqual(result.provider, "openai")
        self.assertIs(result.raw, raw)
