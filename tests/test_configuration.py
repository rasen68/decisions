from dataclasses import FrozenInstanceError, replace
import json
import unittest
from unittest.mock import patch

import httpx

from decisions import AsyncClient, Client
from decisions.providers import Jeff, OpenAI, TypeSafe


def providers():
    return (OpenAI(api_key="synthetic-original"),
            TypeSafe(api_key="synthetic-original"),
            Jeff("http://localhost:8765", api_key="synthetic-original"))


class ProviderConfigurationTests(unittest.TestCase):
    def test_provider_fields_cannot_be_assigned_or_deleted(self):
        for provider in providers():
            for field in ("api_key", "model", "base_url", "name", "path", "key_env"):
                original = getattr(provider, field)
                with self.subTest(provider=provider.name, field=field):
                    with self.assertRaises(FrozenInstanceError):
                        setattr(provider, field, "changed")
                    with self.assertRaises(FrozenInstanceError):
                        delattr(provider, field)
                    self.assertEqual(getattr(provider, field), original)


def assert_read_only(test, client, provider):
    test.assertIs(client.provider, provider)
    for replacement in (None, replace(provider, model="changed"), Jeff("http://other.test")):
        with test.assertRaises(AttributeError):
            client.provider = replacement
        test.assertIs(client.provider, provider)
    with test.assertRaises(AttributeError):
        del client.provider
    test.assertIs(client.provider, provider)


def handler_for(test, provider, requests):
    def handler(request):
        requests.append(request)
        test.assertEqual(str(request.url), provider.base_url.rstrip("/") + provider.path)
        test.assertEqual(request.headers["Authorization"], f"Bearer {provider.api_key}")
        body = json.loads(request.content)
        test.assertEqual(body["model"], provider.model)
        if isinstance(provider, OpenAI):
            test.assertEqual(body["questions"][0]["type"], "predicate")
            answers = [{"type": "predicate", "probability": .9}]
        else:
            test.assertEqual(body["questions"]["answer"]["type"], "noul")
            answers = {"answer": {"type": "noul", "noul": .9}}
        return httpx.Response(200, json={"model": provider.model, "answers": answers,
                                        "usage": {"input_tokens": 1, "output_tokens": 1}})
    return handler


class ClientConfigurationTests(unittest.TestCase):
    def test_provider_is_read_only_and_new_clients_use_new_configuration(self):
        factory = httpx.Client
        for original in providers():
            changed = replace(original, model="changed", api_key="synthetic-new",
                              base_url="http://new.test/v1")
            for provider in (original, changed):
                requests = []
                with self.subTest(provider=provider), patch(
                        "decisions.client.httpx.Client",
                        side_effect=lambda **kwargs: factory(
                            transport=httpx.MockTransport(handler_for(self, provider, requests)),
                            **kwargs)), Client(provider) as client:
                    assert_read_only(self, client, provider)
                    self.assertEqual(client.predicate("Ready?").probability, .9)
                    self.assertEqual(len(requests), 1)


class AsyncClientConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_is_read_only_and_new_clients_use_new_configuration(self):
        factory = httpx.AsyncClient
        for original in providers():
            changed = replace(original, model="changed", api_key="synthetic-new",
                              base_url="http://new.test/v1")
            for provider in (original, changed):
                requests = []
                with self.subTest(provider=provider), patch(
                        "decisions.client.httpx.AsyncClient",
                        side_effect=lambda **kwargs: factory(
                            transport=httpx.MockTransport(handler_for(self, provider, requests)),
                            **kwargs)):
                    async with AsyncClient(provider) as client:
                        assert_read_only(self, client, provider)
                        self.assertEqual((await client.predicate("Ready?")).probability, .9)
                        self.assertEqual(len(requests), 1)
