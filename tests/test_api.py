import json
import unittest
from unittest.mock import patch

import httpx

from decisions.cli.api import DecisionError, request_decision


class ApiTests(unittest.TestCase):
    def test_request_contract_and_raw_fidelity(self):
        questions = [{"type": "predicate", "instructions": "Café?"}]
        raw = '{ "model": "gpt-6-luna", "answers": [{"type":"predicate","name":null,"probability":0.9}], "usage": {"total_tokens": 20} }'
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, content=raw.encode())
        factory = httpx.Client
        with patch("decisions.client.httpx.Client", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
            self.assertEqual(request_decision("évidence", questions, "gpt-6-luna", "synthetic-key"), raw)
        request, = requests
        self.assertEqual(str(request.url), "https://api.openai.com/v1/decisions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.headers["Authorization"], "Bearer synthetic-key")
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(json.loads(request.content), {"input": "évidence", "questions": questions, "model": "gpt-6-luna"})
        self.assertEqual(request.extensions["timeout"]["read"], 60)

    def test_http_errors_show_status_and_api_message(self):
        response = httpx.Response(429, json={"error": {"message": "Rate limit reached"}})
        with patch("decisions.client.Client._send", return_value=response), self.assertRaisesRegex(DecisionError, "429.*Rate limit reached"):
            request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_non_json_http_error(self):
        with patch("decisions.client.Client._send", return_value=httpx.Response(503, text="Service unavailable")), self.assertRaisesRegex(DecisionError, "503.*Service unavailable"):
            request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_network_errors(self):
        for error in (httpx.ConnectError("DNS lookup failed"), httpx.ReadTimeout("timeout")):
            with self.subTest(error=error), patch("decisions.client.Client._send", side_effect=error), self.assertRaises(DecisionError):
                request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_http_protocol_failures(self):
        for error in (httpx.RemoteProtocolError("invalid status"), httpx.ReadError("incomplete body")):
            with self.subTest(error=error), patch("decisions.client.Client._send", side_effect=error), self.assertRaises(DecisionError):
                request_decision("text", [], "gpt-6-luna", "synthetic-key")

    def test_bad_success_response(self):
        for raw in (b"not JSON", b"[]", b"\xff"):
            with self.subTest(raw=raw), patch("decisions.client.Client._send", return_value=httpx.Response(200, content=raw)), self.assertRaises(DecisionError):
                request_decision("text", [], "gpt-6-luna", "synthetic-key")
