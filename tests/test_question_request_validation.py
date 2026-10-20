import asyncio
from collections import UserDict
from types import MappingProxyType
import unittest
from unittest.mock import AsyncMock, patch

from decisions import AsyncClient, Choice, Client, Predicate, Score
from decisions.providers import OpenAI
from decisions.providers.requests import build_request
from test_response_structure import PROVIDERS


def mutated(question, field, value):
    setattr(question, field, value)
    return {"answer": question}


class QuestionRequestValidationTests(unittest.TestCase):
    def assert_no_send(self, make_questions, expected, method="decide", kwargs=None):
        for provider in PROVIDERS:
            for asynchronous in (False, True):
                with self.subTest(provider=provider.name, asynchronous=asynchronous, method=method):
                    if asynchronous:
                        async def run():
                            async with AsyncClient(provider) as client:
                                with patch.object(client, "_send", AsyncMock()) as send:
                                    with self.assertRaises(expected):
                                        await getattr(client, method)(make_questions(), **(kwargs or {}))
                                    send.assert_not_awaited()
                                    send.assert_not_called()
                        asyncio.run(run())
                    else:
                        with Client(provider) as client:
                            with patch.object(client, "_send") as send:
                                with self.assertRaises(expected):
                                    getattr(client, method)(make_questions(), **(kwargs or {}))
                                send.assert_not_called()

    def test_questions_require_a_nonempty_mapping(self):
        for value in (None, [], [Predicate("Ready?")], "question", 1, True):
            with self.subTest(value=value):
                self.assert_no_send(lambda: value, TypeError)
        self.assert_no_send(dict, ValueError)

    def test_names_require_nonblank_strings(self):
        for value in (None, 1, True, b"name"):
            with self.subTest(value=value):
                self.assert_no_send(lambda: {value: Predicate("Ready?")}, TypeError)
        for value in ("", " ", "\n\t", "\u2003"):
            with self.subTest(value=value):
                self.assert_no_send(lambda: {value: Predicate("Ready?")}, ValueError)

    def test_mapping_values_require_question_objects(self):
        for value in (None, "Ready?", {}, 1, True, Predicate):
            with self.subTest(value=value):
                self.assert_no_send(lambda: {"answer": value}, TypeError)

    def test_mutated_question_text_is_revalidated(self):
        for construct in (lambda: Predicate("Ready?"), lambda: Choice("Which?", choices=["a"]), lambda: Score("How much?", levels=["low"])):
            for value, expected in ((None, TypeError), (" \n", ValueError)):
                with self.subTest(construct=construct, value=value):
                    self.assert_no_send(lambda: mutated(construct(), "question", value), expected)

    def test_reassigned_options_are_revalidated(self):
        for field, construct in (("choices", lambda: Choice("Which?", choices=["a"])), ("levels", lambda: Score("How much?", levels=["low"]))):
            for value, expected in ((None, TypeError), (("a",), TypeError), ([], ValueError), ({}, ValueError), (["a", "a"], ValueError), ([1], TypeError), ({"a": None}, TypeError)):
                with self.subTest(field=field, value=value):
                    self.assert_no_send(lambda: mutated(construct(), field, value), expected)

    def test_in_place_option_mutations_are_revalidated(self):
        for kind in ("choice", "score"):
            for operation, expected in (("clear", ValueError), ("duplicate", ValueError), ("bad_value", TypeError), ("description", TypeError)):
                def make():
                    options = {"a": "A"} if operation == "description" else ["a"]
                    question = Choice("Which?", choices=options) if kind == "choice" else Score("How much?", levels=options)
                    if operation == "clear":
                        options.clear()
                    elif operation == "duplicate":
                        options.append("a")
                    elif operation == "bad_value":
                        options.append(None)
                    else:
                        options["a"] = None
                    return {"answer": question}
                with self.subTest(kind=kind, operation=operation):
                    self.assert_no_send(make, expected)

    def test_convenience_methods_reject_bad_inputs_before_sending(self):
        for method, kwargs in (("predicate", {}), ("noul", {}), ("choice", {"choices": ["a"]}), ("score", {"levels": ["low"]})):
            for text, expected in ((None, TypeError), (" ", ValueError)):
                self.assert_no_send(lambda: text, expected, method, kwargs)
        for method, key in (("choice", "choices"), ("score", "levels")):
            self.assert_no_send(lambda: "Question?", ValueError, method, {key: []})
            self.assert_no_send(lambda: "Question?", TypeError, method, {key: [1]})

    def test_valid_mapping_names_text_and_option_order_are_preserved(self):
        for provider in PROVIDERS:
            for wrap in (dict, UserDict, MappingProxyType):
                name = " \n ready \t"
                text = " \n Ready? \t"
                questions = wrap({name: Predicate(text), "choice": Choice("Which?", choices={"b": "B", "a": "A"}), "score": Score("How much?", levels={"high": "Now", "low": "Later"})})
                request = build_request(provider, questions, None, ())
                wire = request["questions"]
                if isinstance(provider, OpenAI):
                    self.assertEqual([entry["name"] for entry in wire], list(questions))
                    self.assertEqual(wire[0]["instructions"], text)
                    self.assertEqual([entry["value"] for entry in wire[1]["choices"]], ["b", "a"])
                    self.assertEqual([entry["label"] for entry in wire[2]["levels"]], ["high", "low"])
                else:
                    self.assertEqual(list(wire), list(questions))
                    self.assertEqual(wire[name]["instructions"], text)
                    self.assertEqual(list(wire["choice"]["criteria"]), ["b", "a"])
                    self.assertEqual(wire["score"]["criteria"], ["Now", "Later"])
                self.assertIs(questions[name].question, text)


if __name__ == "__main__":
    unittest.main()
