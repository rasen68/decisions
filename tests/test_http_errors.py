import unittest
from unittest.mock import patch

import httpx

import decisions
from decisions import AsyncClient, Client, Predicate
from decisions.providers import Jeff, OpenAI, TypeSafe


PROVIDERS = (
    OpenAI(api_key="synthetic-secret"),
    TypeSafe(api_key="synthetic-secret"),
    Jeff("http://localhost:8765"),
)


def error_responses(provider):
    yield 401, {"json": {"error": {"message": "synthetic-secret private-state"}}}
    yield 429, {"content": b"not JSON: synthetic-secret private-state"}
    yield 500, {"content": b""}
    yield 302, {"content": b"redirect"}
    answer = {"type": "predicate", "probability": .9} if provider.name == "openai" else {"type": "noul", "noul": .9}
    answers = [answer] if provider.name == "openai" else {"ready": answer}
    yield 503, {"json": {
        "model": "reported",
        "answers": answers,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }}


class HTTPErrorAssertions:
    def assert_http_error(self, error, provider, status, req_id):
        self.assertIsInstance(error, decisions.HTTPError)
        self.assertIsInstance(error, decisions.DecisionsError)
        self.assertEqual(error.status, status)
        self.assertEqual(error.provider, provider.name)
        self.assertEqual(error.req_id, req_id)
        self.assertIn(str(status), str(error))
        self.assertIn(provider.name, str(error))
        self.assertNotIn("synthetic-secret", str(error))
        self.assertNotIn("private-state", str(error))


class HTTPErrorTests(HTTPErrorAssertions, unittest.TestCase):
    def test_public_exception_hierarchy(self):
        self.assertTrue(issubclass(decisions.DecisionsError, Exception))
        self.assertTrue(issubclass(decisions.HTTPError, decisions.DecisionsError))

    def test_http_failure_precedes_response_decoding(self):
        factory = httpx.Client
        for provider in PROVIDERS:
            for status, body in error_responses(provider):
                for req_id in (None, "synthetic-request-id"):
                    with self.subTest(provider=provider.name, status=status, req_id=req_id):
                        headers = {} if req_id is None else {"x-request-id": req_id}
                        handler = unittest.mock.Mock(return_value=httpx.Response(status, headers=headers, **body))
                        with patch("decisions.client.httpx.Client", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
                            with Client(provider) as client:
                                with self.assertRaises(Exception) as caught:
                                    client.decide({"ready": Predicate("Ready?")}, "private-state")
                        self.assert_http_error(caught.exception, provider, status, req_id)
                        self.assertEqual(handler.call_count, 1)


class AsyncHTTPErrorTests(HTTPErrorAssertions, unittest.IsolatedAsyncioTestCase):
    async def test_http_failure_precedes_response_decoding(self):
        factory = httpx.AsyncClient
        for provider in PROVIDERS:
            for status, body in error_responses(provider):
                for req_id in (None, "synthetic-request-id"):
                    with self.subTest(provider=provider.name, status=status, req_id=req_id):
                        headers = {} if req_id is None else {"x-request-id": req_id}
                        handler = unittest.mock.Mock(return_value=httpx.Response(status, headers=headers, **body))
                        with patch("decisions.client.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
                            async with AsyncClient(provider) as client:
                                with self.assertRaises(Exception) as caught:
                                    await client.decide({"ready": Predicate("Ready?")}, "private-state")
                        self.assert_http_error(caught.exception, provider, status, req_id)
                        self.assertEqual(handler.call_count, 1)
