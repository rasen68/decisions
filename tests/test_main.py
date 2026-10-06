import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import decisions

from decisions.api import DecisionError
from decisions.cli import main

RAW = '{ "model": "gpt-6-luna", "answers": [{"type":"predicate","name":null,"probability":0.9}], "usage": {"total_tokens": 20} }'


class MainTests(unittest.TestCase):
    def run_main(self, arguments, stdin="pipe", key="synthetic-key", response=RAW, error=None):
        output, diagnostics = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO(stdin)), patch.dict("os.environ", {"OPENAI_API_KEY": key}), patch("decisions.cli.request_decision", return_value=response, side_effect=error) as request, contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
            try:
                code = main(arguments)
            except SystemExit as exit:
                code = exit.code
        return code, output.getvalue(), diagnostics.getvalue(), request

    def test_stdin_to_request_and_readable_stdout(self):
        code, output, diagnostics, request = self.run_main(["-p", "Damaged?"])
        self.assertEqual(code, 0)
        self.assertIn("90.0%", output)
        self.assertEqual(diagnostics, "")
        self.assertEqual(request.call_args.args[0], "pipe")

    def test_raw_prints_entire_response_without_reformatting(self):
        code, output, diagnostics, _ = self.run_main(["-p", "Damaged?", "-r", "-i", "text"])
        self.assertEqual(code, 0)
        self.assertEqual(output, RAW + "\n")
        self.assertEqual(diagnostics, "")

    def test_question_file_to_request(self):
        with patch("decisions.cli.load_questions", return_value=[{"type": "predicate", "instructions": "Q"}]) as loader:
            code, _, _, request = self.run_main(["questions.toml", "-i", "text"])
        self.assertEqual(code, 0)
        loader.assert_called_once_with("questions.toml")
        self.assertEqual(request.call_args.args[0], "text")

    def test_key_missing_fails_without_request(self):
        code, output, diagnostics, request = self.run_main(["-p", "Q", "-i", "text"], key="")
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("OPENAI_API_KEY", diagnostics)
        request.assert_not_called()

    def test_request_failure_goes_only_to_stderr(self):
        code, output, diagnostics, _ = self.run_main(["-p", "Q", "-i", "text"], error=DecisionError("HTTP 429: rate limit"))
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("HTTP 429", diagnostics)
        self.assertNotIn("Traceback", diagnostics)

    def test_missing_input_file(self):
        with patch("decisions.files.Path.read_text", side_effect=FileNotFoundError("missing.txt")):
            code, output, diagnostics, request = self.run_main(["missing.txt", "-p", "Q"])
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("missing.txt", diagnostics)
        request.assert_not_called()

    def test_interrupt_returns_130(self):
        code, output, _, _ = self.run_main(["-p", "Q", "-i", "text"], error=KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(output, "")

    def test_question_file_and_input_file_through_http_layer(self):
        root = Path(decisions.__file__).resolve().parent.parent
        response = b'{"answers":[{"type":"predicate","probability":0.95},{"type":"refusal"},{"type":"refusal"}],"usage":{"total_tokens":20}}'
        output = io.StringIO()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-key"}), patch("decisions.api.urlopen", return_value=io.BytesIO(response)) as send, contextlib.redirect_stdout(output):
            code = main([str(root / "examples/questions.toml"), str(root / "examples/questions.json"), "-r"])
        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), response.decode() + "\n")
        import json
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body["input"], (root / "examples/questions.json").read_text())
        self.assertEqual([q["name"] for q in body["questions"]], ["damaged", "severity", "department"])

    def test_invalid_question_file_is_usage_error(self):
        with patch("decisions.files.Path.read_text", return_value="{"):
            code, output, diagnostics, request = self.run_main(["questions.json", "-i", "text"])
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("invalid question file", diagnostics)
        request.assert_not_called()

    def test_broken_pipe_has_no_traceback(self):
        output = Mock()
        output.write.side_effect = BrokenPipeError()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-key"}), patch("decisions.cli.request_decision", return_value=RAW), patch("sys.stdout", output), patch("decisions.cli.silence_broken_pipe") as silence:
            self.assertEqual(main(["-p", "Q", "-i", "text"]), 0)
        silence.assert_called_once()

    def test_refusal_is_successful_and_visible(self):
        code, output, diagnostics, _ = self.run_main(["-p", "Q", "-i", "text"], response='{"answers":[{"type":"refusal","name":null}]}')
        self.assertEqual(code, 0)
        self.assertIn("Refused", output)
        self.assertEqual(diagnostics, "")


if __name__ == "__main__":
    unittest.main()
