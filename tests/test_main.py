import contextlib
import io
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import decisions

from decisions.api import DecisionError
from decisions.cli import main

RAW = '{ "model": "gpt-6-luna", "answers": [{"type":"predicate","name":null,"probability":0.9}], "usage": {"total_tokens": 20} }'


class MainTests(unittest.TestCase):
    def test_images_through_http_layer_and_gate(self):
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1cAAAAASUVORK5CYII=")
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(png)
            for text_args in ([], ["-t", "Inspect this"]):
                with self.subTest(text_args=text_args), patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-key"}), patch("decisions.api.urlopen", return_value=io.BytesIO(RAW.encode())) as send, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = main(["-p", "Damaged?", "-i", str(image), "--eval", "q1 >= .9", "-q"] + text_args)
                self.assertEqual(code, 0)
                content = json.loads(send.call_args.args[0].data)["input"][0]["content"]
                self.assertEqual(len(content), 2 if text_args else 1)
                self.assertEqual(base64.b64decode(content[-1]["image_url"].split(",")[1]), png)
                if text_args:
                    self.assertEqual(content[0], {"type": "input_text", "text": "Inspect this"})

    def test_bad_image_fails_before_request(self):
        for data in (b"", b"plain text", b"<svg/>"):
            with self.subTest(data=data), patch.object(Path, "read_bytes", return_value=data):
                code, output, diagnostics, request = self.run_main(["-p", "Q", "-i", "bad.png"])
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertIn("bad.png", diagnostics)
                request.assert_not_called()
        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError("missing.png")):
            code, output, diagnostics, request = self.run_main(["-p", "Q", "-i", "missing.png", "--eval", "q1 > .5"])
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("missing.png", diagnostics)
        request.assert_not_called()

    def run_main(self, arguments, stdin="pipe", key="synthetic-key", response=RAW, error=None):
        output, diagnostics = io.StringIO(), io.StringIO()
        with patch("sys.stdin", io.StringIO(stdin)), patch.dict("os.environ", {"OPENAI_API_KEY": key}), patch("decisions.cli.request_decision", return_value=response, side_effect=error) as request, contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
            try:
                code = main(arguments)
            except SystemExit as exit:
                code = exit.code
        return code, output.getvalue(), diagnostics.getvalue(), request

    def test_exec_true_uses_quoted_text_and_child_status(self):
        import shlex
        text = "a 'quote'; $(touch unwanted)\nnext line"
        for source, stdin in ((["-t", text], "unused"), ([], text), (["-"], text)):
            with self.subTest(source=source), patch.object(Path, "read_text", return_value=text), patch("decisions.cli.subprocess.run") as run:
                run.return_value.returncode = 7
                code, output, diagnostics, request = self.run_main(["-p", "Q", "--eval", "q1 >= .9", "-x", "echo {}", *source], stdin=stdin)
            self.assertEqual(code, 7)
            self.assertIn("90.0%", output)
            self.assertEqual(diagnostics, "")
            request.assert_called_once()
            run.assert_called_once_with("echo " + shlex.quote(text), shell=True)

    def test_exec_text_file_passes_filename_but_evaluates_content(self):
        import shlex
        filename = "a 'text' file.txt"
        with patch.object(Path, "read_text", return_value="file content") as read, patch("decisions.cli.subprocess.run") as run:
            run.return_value.returncode = 0
            code, output, diagnostics, request = self.run_main([filename, "-p", "Q", "--eval", "q1 >= .9", "-x", "cat -- {}", "-q"])
        self.assertEqual((code, output, diagnostics), (0, "", ""))
        self.assertEqual(request.call_args.args[0], "file content")
        read.assert_called_once_with(encoding="utf-8")
        run.assert_called_once_with("cat -- " + shlex.quote(filename), shell=True)

    def test_exec_quiet_and_image_filename(self):
        import shlex
        filename = "an 'image'.png"
        with patch.object(Path, "read_bytes", return_value=b"\x89PNG\r\n\x1a\n"), patch("decisions.cli.subprocess.run") as run:
            run.return_value.returncode = 0
            code, output, diagnostics, _ = self.run_main(["-p", "Q", "-i", filename, "--eval", "q1 >= .9", "--exec", "inspect {} {}", "-q"])
        self.assertEqual((code, output, diagnostics), (0, "", ""))
        run.assert_called_once_with("inspect " + shlex.quote(filename) + " " + shlex.quote(filename), shell=True)

    def test_exec_never_runs_on_false_or_error(self):
        cases = [
            ("q1 > .9", RAW, None, 1),
            ("q2 > .5", RAW, None, 2),
            ("q1 > .5", '{"answers":[{"type":"refusal"}]}', None, 2),
            ("q1 > .5", '{}', None, 2),
            ("q1 > .5", RAW, DecisionError("API failure"), 2),
        ]
        for expression, response, error, expected in cases:
            with self.subTest(expression=expression, response=response), patch("decisions.cli.subprocess.run") as run:
                code, _, _, _ = self.run_main(["-p", "Q", "--eval", expression, "-x", "echo {}"], response=response, error=error)
            self.assertEqual(code, expected)
            run.assert_not_called()

    def test_exec_start_failure_and_signal(self):
        with patch("decisions.cli.subprocess.run", side_effect=OSError("cannot start shell")):
            code, _, diagnostics, _ = self.run_main(["-p", "Q", "--eval", "q1 > .5", "-x", "echo {}"])
        self.assertEqual(code, 2)
        self.assertIn("cannot start shell", diagnostics)
        with patch("decisions.cli.subprocess.run") as run:
            run.return_value.returncode = -15
            code, _, _, _ = self.run_main(["-p", "Q", "--eval", "q1 > .5", "-x", "echo {}"])
        self.assertEqual(code, 143)

    def test_exec_nul_text_reports_error_without_traceback(self):
        code, _, diagnostics, _ = self.run_main(["-p", "Q", "-t", "a\x00b", "--eval", "q1 > .5", "-x", "echo {}", "-q"])
        self.assertEqual(code, 2)
        self.assertIn("cannot execute", diagnostics)

    def test_exec_real_shell_preserves_text_without_injection(self):
        import shlex
        import sys
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output.txt"
            marker = Path(directory) / "injected"
            text = f"quoted ' text; $(touch {marker})\nsecond line"
            script = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); sys.exit(7)"
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} {shlex.quote(str(target))} {{}}"
            code, output, diagnostics, _ = self.run_main(["-p", "Q", "-t", text, "--eval", "q1 > .5", "-x", command, "-q"])
            self.assertEqual((code, output, diagnostics), (7, "", ""))
            self.assertEqual(target.read_text(), text)
            self.assertFalse(marker.exists())

    def test_gate_true_false_and_quiet(self):
        for expression, expected in (("q1 >= .9", 0), ("q1 > .9", 1)):
            with self.subTest(expression=expression):
                code, output, diagnostics, request = self.run_main(["-p", "Q", "--eval", expression, "--quiet"])
                self.assertEqual(code, expected)
                self.assertEqual(output, "")
                self.assertEqual(diagnostics, "")
                request.assert_called_once()
        code, output, _, _ = self.run_main(["-p", "Q", "--eval", "q1 > .9", "-r"])
        self.assertEqual(code, 1)
        self.assertEqual(output, RAW + "\n")

    def test_named_mixed_gate_and_out_of_domain_choice(self):
        import json
        root = Path(decisions.__file__).resolve().parent.parent
        response = {"answers": [
            {"type": "predicate", "probability": 0.95},
            {"type": "score", "score": 1.5, "confidence": 0.8, "probabilities": [
                {"value": 0, "label": "low", "probability": 0.1},
                {"value": 1, "label": "medium", "probability": 0.3},
                {"value": 2, "label": "high", "probability": 0.6}]},
            {"type": "choice", "choice": "support", "confidence": 0.9, "probabilities": [
                {"value": "support", "probability": 0.9}, {"value": "billing", "probability": 0.1}]},
        ]}
        arguments = [str(root / "examples/questions.toml"), "--eval",
                     'damaged >= .9 and severity >= 1.5 and severity.confidence >= .8 and department == "support" and department.confidence >= .9', "-q"]
        code, output, diagnostics, request = self.run_main(arguments, response=json.dumps(response))
        self.assertEqual((code, output, diagnostics), (0, "", ""))
        request.assert_called_once()
        self.assertEqual(len(request.call_args.args[1]), 3)
        response["answers"][2]["choice"] = "unknown"
        code, output, diagnostics, _ = self.run_main(arguments, response=json.dumps(response))
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("outside the supplied choices", diagnostics)

    def test_non_selected_choice_probability_gate(self):
        response = '{"answers":[{"type":"choice","choice":"support","confidence":0.8,"probabilities":[{"value":"support","probability":0.7},{"value":"billing","probability":0.3}]}]}'
        arguments = ["-c", "Department?", "billing", "support", "--eval", 'q1["billing"] >= .3', "-q"]
        code, output, diagnostics, request = self.run_main(arguments, response=response)
        self.assertEqual((code, output, diagnostics), (0, "", ""))
        request.assert_called_once()
        arguments[-2] = 'q1 == "support" or q1["billing"] >= .3'
        missing = '{"answers":[{"type":"choice","choice":"support","confidence":0.8,"probabilities":[{"value":"support","probability":1}]}]}'
        code, output, diagnostics, _ = self.run_main(arguments, response=missing)
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("probability", diagnostics)
        arguments[-2] = 'q1["missing"] >= .3'
        code, _, _, request = self.run_main(arguments, response=response)
        self.assertEqual(code, 2)
        request.assert_not_called()

    def test_gate_invalid_expression_never_sends_request(self):
        for expression in ("q2 > .5", "q1.confidence > .5", "q1 > .5 or unknown > 0"):
            code, output, diagnostics, request = self.run_main(["-p", "Q", "--eval", expression])
            self.assertEqual(code, 2)
            self.assertEqual(output, "")
            self.assertIn("expression", diagnostics)
            request.assert_not_called()

    def test_gate_errors_and_refusals_cannot_pass_through_negation(self):
        for response in ('{"answers":[{"type":"refusal"}]}', '{"answers":[{"type":"predicate","probability":2}]}', '{}'):
            code, output, diagnostics, _ = self.run_main(["-p", "Q", "--eval", "not (q1 > .5)", "--quiet"], response=response)
            self.assertEqual(code, 2)
            self.assertEqual(output, "")
            self.assertTrue(diagnostics)
        code, _, _, _ = self.run_main(["-p", "Q", "--eval", "q1 > .5"], error=DecisionError("HTTP 429"))
        self.assertEqual(code, 2)

    def test_gate_checks_unreferenced_answers_before_short_circuiting(self):
        response = '{"answers":[{"type":"predicate","probability":0.9},{"type":"refusal"}]}'
        code, output, diagnostics, request = self.run_main(["-p", "Q", "-p", "Other", "--eval", "q1 >= .9 or q2 > .5", "--quiet"], response=response)
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("refused", diagnostics)
        request.assert_called_once()

    def test_stdin_to_request_and_readable_stdout(self):
        code, output, diagnostics, request = self.run_main(["-p", "Damaged?"])
        self.assertEqual(code, 0)
        self.assertIn("90.0%", output)
        self.assertEqual(diagnostics, "")
        self.assertEqual(request.call_args.args[0], "pipe")

    def test_raw_prints_entire_response_without_reformatting(self):
        code, output, diagnostics, _ = self.run_main(["-p", "Damaged?", "-r", "-t", "text"])
        self.assertEqual(code, 0)
        self.assertEqual(output, RAW + "\n")
        self.assertEqual(diagnostics, "")

    def test_graphic_output_through_cli(self):
        for flag in ("-g", "--graphic"):
            with self.subTest(flag=flag):
                code, output, diagnostics, _ = self.run_main([flag, "-p", "Damaged?", "-t", "text"])
                self.assertEqual(code, 0)
                self.assertEqual(output, "1. Damaged?\n  Probability: [██████████████████░░] 90.0%\n")
                self.assertEqual(diagnostics, "")

    def test_question_file_to_request(self):
        with patch("decisions.cli.load_questions", return_value=[{"type": "predicate", "instructions": "Q"}]) as loader:
            code, _, _, request = self.run_main(["questions.toml", "-t", "text"])
        self.assertEqual(code, 0)
        loader.assert_called_once_with("questions.toml")
        self.assertEqual(request.call_args.args[0], "text")

    def test_key_missing_fails_without_request(self):
        code, output, diagnostics, request = self.run_main(["-p", "Q", "-t", "text"], key="")
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("OPENAI_API_KEY", diagnostics)
        request.assert_not_called()

    def test_request_failure_goes_only_to_stderr(self):
        code, output, diagnostics, _ = self.run_main(["-p", "Q", "-t", "text"], error=DecisionError("HTTP 429: rate limit"))
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
        code, output, _, _ = self.run_main(["-p", "Q", "-t", "text"], error=KeyboardInterrupt())
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
            code, output, diagnostics, request = self.run_main(["questions.json", "-t", "text"])
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("invalid question file", diagnostics)
        request.assert_not_called()

    def test_broken_pipe_has_no_traceback(self):
        output = Mock()
        output.write.side_effect = BrokenPipeError()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-key"}), patch("decisions.cli.request_decision", return_value=RAW), patch("sys.stdout", output), patch("decisions.cli.silence_broken_pipe") as silence:
            self.assertEqual(main(["-p", "Q", "-t", "text"]), 0)
        silence.assert_called_once()

    def test_refusal_is_successful_and_visible(self):
        code, output, diagnostics, _ = self.run_main(["-p", "Q", "-t", "text"], response='{"answers":[{"type":"refusal","name":null}]}')
        self.assertEqual(code, 0)
        self.assertIn("Refused", output)
        self.assertEqual(diagnostics, "")


if __name__ == "__main__":
    unittest.main()
