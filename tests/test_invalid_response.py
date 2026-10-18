import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

import decisions
from decisions import AsyncClient, Client, Predicate
from decisions.providers import Jeff, OpenAI, TypeSafe
from decisions.types import InvalidResponseError


PROVIDERS = (
    OpenAI(api_key="synthetic-secret"),
    TypeSafe(api_key="synthetic-secret"),
    Jeff("http://localhost:8765", api_key="synthetic-secret"),
)
QUESTIONS = {"answer": Predicate("Is it ready?")}
INVALID_BODIES = (b"not json", b"", b" \n\t", b'{"private-state":')


class InvalidResponseAssertions:
    def assert_invalid_response(self, error, provider):
        self.assertIsInstance(error, decisions.DecisionsError)
        self.assertIsInstance(error.__cause__, json.JSONDecodeError)
        self.assertIn(provider.name, str(error))
        for private in ("synthetic-secret", "private-state"):
            self.assertNotIn(private, str(error))


class InvalidResponseTests(InvalidResponseAssertions, unittest.TestCase):
    def test_public_exception_exports_and_hierarchy(self):
        self.assertIs(decisions.InvalidResponseError, InvalidResponseError)
        self.assertTrue(issubclass(InvalidResponseError, decisions.DecisionsError))

    def test_invalid_json_is_invalid_response(self):
        for provider in PROVIDERS:
            for content in INVALID_BODIES:
                with self.subTest(provider=provider.name, content=content):
                    response = httpx.Response(200, content=content)
                    with Client(provider) as client:
                        with patch.object(client, "_send", return_value=response) as send:
                            with self.assertRaises(InvalidResponseError) as caught:
                                client.decide(QUESTIONS, "private-state")
                    send.assert_called_once()
                    self.assert_invalid_response(caught.exception, provider)

    def test_json_error_cause_is_preserved(self):
        original = json.JSONDecodeError("synthetic-secret", "private-state", 0)
        response = httpx.Response(200)
        with Client(PROVIDERS[0]) as client:
            with patch.object(client, "_send", return_value=response):
                with patch.object(response, "json", side_effect=original):
                    with self.assertRaises(InvalidResponseError) as caught:
                        client.decide(QUESTIONS, "private-state")
        self.assertIs(caught.exception.__cause__, original)
        self.assert_invalid_response(caught.exception, PROVIDERS[0])

    def test_http_error_precedes_json_decoding(self):
        response = httpx.Response(503, content=b"not json")
        with Client(PROVIDERS[0]) as client:
            with patch.object(client, "_send", return_value=response) as send:
                with patch.object(response, "json") as decode:
                    with self.assertRaises(decisions.HTTPError) as caught:
                        client.decide(QUESTIONS)
        self.assertEqual(caught.exception.status, 503)
        send.assert_called_once()
        decode.assert_not_called()

    def test_decoder_programming_errors_propagate(self):
        for error_type in (ValueError, KeyError, TypeError, IndexError, RuntimeError):
            with self.subTest(error=error_type):
                original = error_type("programming mistake")
                response = httpx.Response(200, json={})
                with Client(PROVIDERS[0]) as client:
                    with patch.object(client, "_send", return_value=response):
                        with patch("decisions.client.decode_result", side_effect=original):
                            with self.assertRaises(error_type) as caught:
                                client.decide(QUESTIONS)
                self.assertIs(caught.exception, original)

    def test_valid_json_reaches_decoder_unchanged(self):
        raw = {"unknown_metadata": [1, 2]}
        result = object()
        response = httpx.Response(200)
        with Client(PROVIDERS[0]) as client:
            with patch.object(client, "_send", return_value=response):
                with patch.object(response, "json", return_value=raw):
                    with patch("decisions.client.decode_result", return_value=result) as decode:
                        self.assertIs(client.decide(QUESTIONS), result)
        decode.assert_called_once_with(PROVIDERS[0], QUESTIONS, raw)
        self.assertIs(decode.call_args.args[2], raw)


class AsyncInvalidResponseTests(
    InvalidResponseAssertions, unittest.IsolatedAsyncioTestCase
):
    async def test_invalid_json_is_invalid_response(self):
        for provider in PROVIDERS:
            for content in INVALID_BODIES:
                with self.subTest(provider=provider.name, content=content):
                    response = httpx.Response(200, content=content)
                    async with AsyncClient(provider) as client:
                        with patch.object(client, "_send", AsyncMock(return_value=response)) as send:
                            with self.assertRaises(InvalidResponseError) as caught:
                                await client.decide(QUESTIONS, "private-state")
                    send.assert_awaited_once()
                    self.assert_invalid_response(caught.exception, provider)

    async def test_json_error_cause_is_preserved(self):
        original = json.JSONDecodeError("synthetic-secret", "private-state", 0)
        response = httpx.Response(200)
        async with AsyncClient(PROVIDERS[0]) as client:
            with patch.object(client, "_send", AsyncMock(return_value=response)):
                with patch.object(response, "json", side_effect=original):
                    with self.assertRaises(InvalidResponseError) as caught:
                        await client.decide(QUESTIONS, "private-state")
        self.assertIs(caught.exception.__cause__, original)
        self.assert_invalid_response(caught.exception, PROVIDERS[0])

    async def test_http_error_precedes_json_decoding(self):
        response = httpx.Response(503, content=b"not json")
        async with AsyncClient(PROVIDERS[0]) as client:
            with patch.object(client, "_send", AsyncMock(return_value=response)) as send:
                with patch.object(response, "json") as decode:
                    with self.assertRaises(decisions.HTTPError) as caught:
                        await client.decide(QUESTIONS)
        self.assertEqual(caught.exception.status, 503)
        send.assert_awaited_once()
        decode.assert_not_called()

    async def test_decoder_programming_errors_propagate(self):
        for error_type in (ValueError, KeyError, TypeError, IndexError, RuntimeError):
            with self.subTest(error=error_type):
                original = error_type("programming mistake")
                response = httpx.Response(200, json={})
                async with AsyncClient(PROVIDERS[0]) as client:
                    with patch.object(client, "_send", AsyncMock(return_value=response)):
                        with patch("decisions.client.decode_result", side_effect=original):
                            with self.assertRaises(error_type) as caught:
                                await client.decide(QUESTIONS)
                self.assertIs(caught.exception, original)

    async def test_valid_json_reaches_decoder_unchanged(self):
        raw = {"unknown_metadata": [1, 2]}
        result = object()
        response = httpx.Response(200)
        async with AsyncClient(PROVIDERS[0]) as client:
            with patch.object(client, "_send", AsyncMock(return_value=response)):
                with patch.object(response, "json", return_value=raw):
                    with patch("decisions.client.decode_result", return_value=result) as decode:
                        self.assertIs(await client.decide(QUESTIONS), result)
        decode.assert_called_once_with(PROVIDERS[0], QUESTIONS, raw)
        self.assertIs(decode.call_args.args[2], raw)


if __name__ == "__main__":
    unittest.main()
