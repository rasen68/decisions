import unittest
from unittest.mock import patch

from decisions import AsyncClient, Client
from decisions.providers import Jeff, OpenAI, TypeSafe


def providers(key):
    return (OpenAI(api_key=key), TypeSafe(api_key=key),
            Jeff("http://localhost:8765", api_key=key))


class CredentialTests(unittest.TestCase):
    def assert_rejected(self, provider, environment, error):
        for client_type in (Client, AsyncClient):
            with self.subTest(client=client_type.__name__, provider=provider.name if provider else "default"), \
                    patch.dict("os.environ", environment, clear=True), \
                    patch("decisions.client.httpx.Client") as sync_http, \
                    patch("decisions.client.httpx.AsyncClient") as async_http:
                with self.assertRaises(error) as caught:
                    client_type(provider)
                self.assertNotIn("synthetic-secret", str(caught.exception))
                self.assertNotIn("synthetic-secret", repr(caught.exception))
                sync_http.assert_not_called()
                async_http.assert_not_called()

    def test_hosted_keys_are_required(self):
        for provider in (OpenAI(), TypeSafe()):
            for value in (None, "", " \t\n"):
                environment = {} if value is None else {provider.key_env: value}
                self.assert_rejected(provider, environment, ValueError)

    def test_invalid_explicit_key_does_not_fall_back(self):
        environment = {"OPENAI_API_KEY": "synthetic-env", "TYPESAFE_API_KEY": "synthetic-env"}
        for key in ("", " \t\n", 123, False, b"synthetic-secret", ["synthetic-secret"]):
            for provider in providers(key):
                self.assert_rejected(provider, environment,
                                     ValueError if isinstance(key, str) else TypeError)

    def test_resolved_keys_and_optional_jeff_auth(self):
        environment = {"OPENAI_API_KEY": "synthetic-env", "TYPESAFE_API_KEY": "synthetic-env"}
        cases = [(provider, "Bearer  synthetic-explicit ")
                 for provider in providers(" synthetic-explicit ")]
        cases.extend([(OpenAI(), "Bearer synthetic-env"),
                      (TypeSafe(), "Bearer synthetic-env"),
                      (Jeff("http://localhost:8765"), None),
                      (None, "Bearer synthetic-env")])
        for provider, authorization in cases:
            for client_type, http_name in ((Client, "Client"), (AsyncClient, "AsyncClient")):
                with self.subTest(client=client_type.__name__, provider=provider), \
                        patch.dict("os.environ", environment, clear=True), \
                        patch(f"decisions.client.httpx.{http_name}") as http:
                    client_type(provider)
                    http.assert_called_once()
                    headers = http.call_args.kwargs["headers"]
                    self.assertEqual(headers.get("Authorization"), authorization)
                    self.assertEqual(headers["Accept"], "application/json")

    def test_default_provider_requires_key(self):
        self.assert_rejected(None, {}, ValueError)

    def test_provider_repr_hides_key(self):
        for provider in providers("synthetic-secret"):
            with self.subTest(provider=provider.name):
                self.assertNotIn("synthetic-secret", repr(provider))
                self.assertNotIn("api_key=", repr(provider))
                self.assertIn(provider.model, repr(provider))
                self.assertEqual(provider.api_key, "synthetic-secret")
