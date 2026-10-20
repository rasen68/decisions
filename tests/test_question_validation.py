from collections import UserDict
from types import MappingProxyType
import unittest

from decisions import Choice, Noul, Predicate, Score


def constructors(text):
    return (
        lambda: Predicate(text),
        lambda: Noul(text),
        lambda: Choice(text, choices=["a"]),
        lambda: Score(text, levels=["low"]),
    )


class QuestionValidationTests(unittest.TestCase):
    def test_question_text_requires_a_string(self):
        for text in (None, 1, True, [], {}, b"Ready?"):
            for construct in constructors(text):
                with self.subTest(text=text, construct=construct):
                    with self.assertRaises(TypeError):
                        construct()

    def test_question_text_must_not_be_blank(self):
        for text in ("", " ", "\n\t", "\u2003"):
            for construct in constructors(text):
                with self.subTest(text=text, construct=construct):
                    with self.assertRaises(ValueError):
                        construct()

    def test_valid_text_is_preserved(self):
        text = " \n Ready? \t"
        for construct in constructors(text):
            self.assertIs(construct().question, text)

    def test_options_require_a_list_or_mapping(self):
        for value in (None, "abc", b"abc", ("a", "b"), {"a", "b"}, 3, True):
            for construct in (lambda: Choice("Which?", choices=value), lambda: Score("How much?", levels=value)):
                with self.subTest(value=value):
                    with self.assertRaises(TypeError):
                        construct()

    def test_options_must_not_be_empty(self):
        for value in ([], {}, UserDict()):
            for construct in (lambda: Choice("Which?", choices=value), lambda: Score("How much?", levels=value)):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        construct()

    def test_choice_values_require_strings_or_booleans(self):
        for value in (None, 0, 1, 1.0, [], {}, b"a"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    Choice("Which?", choices=["a", value])
        for value in (None, 0, 1, 1.0, b"a"):
            with self.subTest(key=value):
                with self.assertRaises(TypeError):
                    Choice("Which?", choices={value: "Description"})

    def test_level_labels_require_strings(self):
        for value in (None, True, False, 0, 1.0, [], {}, b"low"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    Score("How much?", levels=["low", value])
        for value in (None, True, 0, 1.0, b"low"):
            with self.subTest(key=value):
                with self.assertRaises(TypeError):
                    Score("How much?", levels={value: "Description"})

    def test_descriptions_require_strings(self):
        for value in (None, True, 1, [], {}, b"Description"):
            for construct in (lambda: Choice("Which?", choices={"a": value}), lambda: Score("How much?", levels={"low": value})):
                with self.subTest(value=value):
                    with self.assertRaises(TypeError):
                        construct()

    def test_duplicate_options_are_rejected(self):
        for choices in (["a", "a"], [True, True], [False, False], [True, "true", True]):
            with self.subTest(choices=choices):
                with self.assertRaises(ValueError):
                    Choice("Which?", choices=choices)
        with self.assertRaises(ValueError):
            Score("How much?", levels=["low", "high", "low"])

    def test_valid_options_preserve_identity_order_and_public_types(self):
        for options in ([True, "true", False, "false", ""], {True: "Yes", "true": "Literal", False: "No"}, UserDict({"b": "", "a": "A"}), MappingProxyType({"b": "B", "a": "A"})):
            with self.subTest(options=options):
                question = Choice("Which?", choices=options)
                self.assertIs(question.choices, options)
                self.assertEqual(list(question.choices), list(options))
        for options in (["high", "low", ""], {"high": "", "low": "Later"}, UserDict({"high": "Now"})):
            self.assertIs(Score("How much?", levels=options).levels, options)


if __name__ == "__main__":
    unittest.main()
