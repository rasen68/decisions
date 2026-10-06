import io
from http.client import BadStatusLine, IncompleteRead
import json
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from decisions.api import DecisionError, request_decision


class ApiTests(unittest.TestCase):
    def test_request_contract_and_raw_fidelity(self):
        questions = [{"type": "predicate", "instructions": "Café?"}]
        raw = '{ "model": "gpt-6-luna", "answers": [{"type":"predicate","name":null,"probability":0.9}], "usage": {"total_tokens": 20} }'
        with patch("decisions.api.urlopen", return_value=io.BytesIO(raw.encode())) as send:
            self.assertEqual(request_decision("évidence", questions, "gpt-6-luna", "synthetic-key"), raw)
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/decisions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(request.data), {"input": "évidence", "questions": questions, "model": "gpt-6-luna"})
        self.assertEqual(send.call_args.kwargs["timeout"], 60)

    def test_http_errors_show_status_and_api_message(self):
        error = HTTPError("https://api.openai.com/v1/decisions", 429, "Too Many Requests", {}, io.BytesIO(b'{"error":{"message":"Rate limit reached"}}'))
        with patch("decisions.api.urlopen", side_effect=error), self.assertRaisesRegex(DecisionError, "429.*Rate limit reached"):
            request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_non_json_http_error(self):
        error = HTTPError("https://api.openai.com/v1/decisions", 503, "Unavailable", {}, io.BytesIO(b"Service unavailable"))
        with patch("decisions.api.urlopen", side_effect=error), self.assertRaisesRegex(DecisionError, "503.*Service unavailable"):
            request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_network_errors(self):
        for error in (URLError("DNS lookup failed"), TimeoutError()):
            with self.subTest(error=error), patch("decisions.api.urlopen", side_effect=error), self.assertRaises(DecisionError):
                request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_http_protocol_failures(self):
        for error in (BadStatusLine("invalid status"), IncompleteRead(b"partial", 20)):
            with self.subTest(error=error), patch("decisions.api.urlopen", side_effect=error), self.assertRaises(DecisionError):
                request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_error_body_protocol_failure_still_reports_status(self):
        class TruncatedBody(io.BytesIO):
            def read(self, *args):
                raise IncompleteRead(b"partial", 20)
        error = HTTPError("https://api.openai.com/v1/decisions", 503, "Unavailable", {}, TruncatedBody())
        with patch("decisions.api.urlopen", side_effect=error), self.assertRaisesRegex(DecisionError, "503"):
            request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_bad_success_response(self):
        for raw in (b"not JSON", b"[]", b"\xff"):
            with self.subTest(raw=raw), patch("decisions.api.urlopen", return_value=io.BytesIO(raw)), self.assertRaises(DecisionError):
                request_decision("text", [], "gpt-6-luna", "synthetic-key")


if __name__ == "__main__":
    unittest.main()
