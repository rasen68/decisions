import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

import decisions
from decisions import AsyncClient, Client, Predicate
from decisions.providers import Jeff, OpenAI, TypeSafe


PROVIDERS = (
    OpenAI(api_key="synthetic-secret"),
    TypeSafe(api_key="synthetic-secret"),
    Jeff("http://localhost:8765"),
)


class TransportErrorAssertions:
    def assert_transport_error(self, error, original, provider, expected):
        self.assertIsInstance(error, expected)
        self.assertIsInstance(error, decisions.DecisionsError)
        self.assertIs(error.__cause__, original)
        self.assertIn(provider.name, str(error))
        for private in ("synthetic-secret", "private-state", "https://"):
            self.assertNotIn(private, str(error))


class ConnectionErrorTests(TransportErrorAssertions, unittest.TestCase):
    def test_public_hierarchy(self):
        self.assertTrue(issubclass(
            decisions.ConnectionError, decisions.DecisionsError
        ))
        self.assertTrue(issubclass(
            decisions.TimeoutError, decisions.ConnectionError
        ))

    def test_timeouts(self):
        for error_type in (
            httpx.ConnectTimeout, httpx.ReadTimeout,
            httpx.WriteTimeout, httpx.PoolTimeout,
        ):
            with self.subTest(error=error_type):
                original = error_type("synthetic-secret private-state")
                provider = PROVIDERS[0]
                with Client(provider) as client:
                    with patch.object(client, "_send", side_effect=original) as send:
                        with self.assertRaises(Exception) as caught:
                            client.predicate("Ready?", "Text")
                self.assert_transport_error(
                    caught.exception, original, provider, decisions.TimeoutError
                )
                send.assert_called_once()

    def test_connection_failures(self):
        for provider in PROVIDERS:
            for error_type in (
                httpx.ConnectError, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.RequestError,
            ):
                with self.subTest(provider=provider.name, error=error_type):
                    original = error_type(
                        "https://synthetic-secret.test/private-state"
                    )
                    with Client(provider) as client:
                        with patch.object(client, "_send", Mock(side_effect=original)) as send:
                            with self.assertRaises(Exception) as caught:
                                client.decide(
                                    {"ready": Predicate("Ready?")}, "private-state"
                                )
                    self.assert_transport_error(
                        caught.exception, original, provider,
                        decisions.ConnectionError,
                    )
                    send.assert_called_once()

    def test_programming_errors_are_not_wrapped(self):
        original = TypeError("programming mistake")
        with Client(PROVIDERS[0]) as client:
            with patch.object(client, "_send", side_effect=original):
                with self.assertRaises(TypeError) as caught:
                    client.predicate("Ready?", "Text")
        self.assertIs(caught.exception, original)


class AsyncConnectionErrorTests(
    TransportErrorAssertions, unittest.IsolatedAsyncioTestCase
):
    async def test_cancellation_propagates(self):
        original = asyncio.CancelledError()
        async with AsyncClient(PROVIDERS[0]) as client:
            with patch.object(client, "_send", AsyncMock(side_effect=original)) as send:
                with self.assertRaises(asyncio.CancelledError) as caught:
                    await client.predicate("Ready?", "Text")
        self.assertIs(caught.exception, original)
        send.assert_awaited_once()

    async def test_timeouts(self):
        for error_type in (
            httpx.ConnectTimeout, httpx.ReadTimeout,
            httpx.WriteTimeout, httpx.PoolTimeout,
        ):
            with self.subTest(error=error_type):
                original = error_type("synthetic-secret private-state")
                provider = PROVIDERS[0]
                async with AsyncClient(provider) as client:
                    with patch.object(client, "_send", AsyncMock(side_effect=original)) as send:
                        with self.assertRaises(Exception) as caught:
                            await client.predicate("Ready?", "Text")
                self.assert_transport_error(
                    caught.exception, original, provider, decisions.TimeoutError
                )
                send.assert_awaited_once()

    async def test_connection_failures(self):
        for provider in PROVIDERS:
            for error_type in (
                httpx.ConnectError, httpx.ReadError,
                httpx.RemoteProtocolError, httpx.RequestError,
            ):
                with self.subTest(provider=provider.name, error=error_type):
                    original = error_type(
                        "https://synthetic-secret.test/private-state"
                    )
                    async with AsyncClient(provider) as client:
                        with patch.object(client, "_send", AsyncMock(side_effect=original)) as send:
                            with self.assertRaises(Exception) as caught:
                                await client.decide(
                                    {"ready": Predicate("Ready?")}, "private-state"
                                )
                    self.assert_transport_error(
                        caught.exception, original, provider,
                        decisions.ConnectionError,
                    )
                    send.assert_awaited_once()

    async def test_programming_errors_are_not_wrapped(self):
        original = TypeError("programming mistake")
        async with AsyncClient(PROVIDERS[0]) as client:
            with patch.object(client, "_send", AsyncMock(side_effect=original)):
                with self.assertRaises(TypeError) as caught:
                    await client.predicate("Ready?", "Text")
        self.assertIs(caught.exception, original)
