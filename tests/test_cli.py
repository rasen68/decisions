import contextlib
import io
import unittest

from decisions.cli import parse_arguments


class ArgumentsTests(unittest.TestCase):
    def parse(self, *arguments):
        return parse_arguments(list(arguments))

    def test_mixed_repeated_questions_preserve_order(self):
        args = self.parse("input.txt", "-p", "Damaged?", "-s", "Severity?", "low", "high: Fully blocked", "-c", "Department?", "billing", "support", "-p", "Urgent?")
        self.assertEqual(args.input_file, "input.txt")
        self.assertEqual([q["type"] for q in args.questions], ["predicate", "score", "choice", "predicate"])
        self.assertEqual([q["instructions"] for q in args.questions], ["Damaged?", "Severity?", "Department?", "Urgent?"])
        self.assertEqual(args.questions[1]["levels"], [{"label": "low", "description": "low"}, {"label": "high", "description": "Fully blocked"}])
        self.assertEqual(args.questions[2]["choices"], [{"value": "billing", "description": "billing"}, {"value": "support", "description": "support"}])

    def test_input_after_separator(self):
        args = self.parse("-s", "Severity?", "low", "high", "--", "input.txt")
        self.assertEqual(args.input_file, "input.txt")
        self.assertEqual(len(args.questions[0]["levels"]), 2)

    def test_question_file_and_input_file(self):
        args = self.parse("questions.toml", "input.txt")
        self.assertEqual(args.question_file, "questions.toml")
        self.assertEqual(args.input_file, "input.txt")

    def test_inline_input_and_model(self):
        args = self.parse("-p", "Damaged?", "-i", "Broken screen", "--model", "future-model", "-r")
        self.assertEqual(args.input, "Broken screen")
        self.assertEqual(args.model, "future-model")
        self.assertTrue(args.raw)

    def test_split_only_first_colon(self):
        args = self.parse("-c", "Department?", "billing : Payments: refunds", "support")
        self.assertEqual(args.questions[0]["choices"][0], {"value": "billing", "description": "Payments: refunds"})

    def test_bad_arguments_exit_two(self):
        cases = [[], ["-s", "Severity?"], ["-c", "Department?"], ["-p", ""], ["-c", "Department?", ": description", "other"], ["-c", "Department?", "same", "same"], ["input.txt", "-p", "Damaged?", "-i", "text"], ["questions.json", "input.txt", "-p", "Damaged?"], ["-p", "Damaged?", "--mod", "other"], ["one.json", "two.txt", "three.txt"]]
        for arguments in cases:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse(*arguments)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
