import io
import unittest
from pathlib import Path
from unittest.mock import patch

import decisions
from decisions.cli import parse_arguments
from decisions.files import load_questions, read_input
from decisions.questions import QuestionError, validate_questions

ROOT = Path(decisions.__file__).resolve().parent.parent


class FilesTests(unittest.TestCase):
    def test_json_and_toml_have_same_questions(self):
        json_questions = load_questions(str(ROOT / "examples/questions.json"))
        toml_questions = load_questions(str(ROOT / "examples/questions.toml"))
        self.assertEqual(json_questions, toml_questions)
        self.assertEqual([q["name"] for q in json_questions], ["damaged", "severity", "department"])

    def test_boolean_values_remain_typed(self):
        questions = [{"type": "choice", "instructions": "Which?", "choices": [{"value": True}, {"value": "true"}, {"value": False}]}]
        self.assertIs(validate_questions(questions), questions)
        self.assertIs(questions[0]["choices"][0]["value"], True)

    def test_invalid_schema(self):
        cases = [None, [], ["question"], [{}], [{"type": "other", "instructions": "Q"}], [{"type": "predicate", "instructions": ""}], [{"type": "predicate", "instructions": "Q", "levels": []}], [{"type": "predicate", "instructions": "Q", "name": None}], [{"type": "score", "instructions": "Q", "levels": []}], [{"type": "score", "instructions": "Q", "levels": [{"value": "low"}]}], [{"type": "choice", "instructions": "Q", "choices": [{"value": 42}]}], [{"type": "choice", "instructions": "Q", "choices": [{"value": "a", "description": 42}]}], [{"type": "choice", "instructions": "Q", "choices": [{"value": "a"}, {"value": "a"}]}], [{"type": "predicate", "instructions": "Q", "name": "a"}, {"type": "predicate", "instructions": "Q2", "name": "a"}]]
        for questions in cases:
            with self.subTest(questions=questions), self.assertRaises(QuestionError):
                validate_questions(questions)

    def test_bad_file_contents(self):
        cases = [("q.yaml", "questions: []"), ("q.json", "{"), ("q.toml", "questions = ["), ("q.json", "[]"), ("q.json", '{"questions": [], "input": "text"}')]
        for filename, content in cases:
            with self.subTest(filename=filename, content=content), patch.object(Path, "read_text", return_value=content):
                with self.assertRaises(QuestionError):
                    load_questions(filename)

    def test_input_argument_preserves_whitespace_and_beats_pipe(self):
        args = parse_arguments(["-p", "Q", "-i", "  text\n"])
        self.assertEqual(read_input(args, io.StringIO("ignored")), "  text\n")

    def test_input_file_is_always_text_even_if_json(self):
        args = parse_arguments(["data.json", "-p", "Q"])
        with patch.object(Path, "read_text", return_value='{"text": "value"}\n') as reader:
            self.assertEqual(read_input(args, io.StringIO("ignored")), '{"text": "value"}\n')
        reader.assert_called_once_with(encoding="utf-8")

    def test_stdin_implicit_and_explicit(self):
        for argv in (["-p", "Q"], ["-p", "Q", "--", "-"]):
            with self.subTest(argv=argv):
                self.assertEqual(read_input(parse_arguments(argv), io.StringIO("pipe\n")), "pipe\n")

    def test_missing_terminal_input_has_usage_error(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        with self.assertRaises(QuestionError):
            read_input(parse_arguments(["-p", "Q"]), stream)

    def test_explicit_dash_can_read_terminal(self):
        stream = io.StringIO("typed text")
        stream.isatty = lambda: True
        self.assertEqual(read_input(parse_arguments(["-", "-p", "Q"]), stream), "typed text")


if __name__ == "__main__":
    unittest.main()
