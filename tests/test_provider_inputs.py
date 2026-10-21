import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from decisions import AsyncClient, Choice, Client, Predicate
from decisions.providers import OpenAI
from decisions.providers.requests import build_request
from test_response_structure import PROVIDERS


class ProviderInputTests(unittest.TestCase):
    def assert_no_send(self, provider, method, argument, kwargs, expected):
        for asynchronous in (False, True):
            with self.subTest(provider=provider.name, method=method, asynchronous=asynchronous):
                if asynchronous:
                    async def run():
                        async with AsyncClient(provider) as client:
                            with patch.object(client, "_send", AsyncMock()) as send:
                                with self.assertRaises(expected):
                                    await getattr(client, method)(argument, **kwargs)
                                send.assert_not_called()
                    asyncio.run(run())
                else:
                    with Client(provider) as client:
                        with patch.object(client, "_send") as send:
                            with self.assertRaises(expected):
                                getattr(client, method)(argument, **kwargs)
                            send.assert_not_called()

    def test_systemone_boolean_choices_are_rejected(self):
        for provider in PROVIDERS[1:]:
            for options in ([True, False], [True, "true"], {False: "No", "false": "Literal"}):
                self.assert_no_send(provider, "decide", {"answer": Choice("Which?", choices=options)}, {}, ValueError)
                self.assert_no_send(provider, "choice", "Which?", {"choices": options}, ValueError)

    def test_mutated_boolean_options_are_rejected(self):
        for provider in PROVIDERS[1:]:
            for options in (["a"], {"a": "A"}):
                question = Choice("Which?", choices=options)
                if isinstance(options, list):
                    options.append(True)
                else:
                    options[False] = "No"
                self.assert_no_send(provider, "decide", {"ready": Predicate("Ready?"), "answer": question}, {}, ValueError)

    def test_supported_choice_values_are_preserved(self):
        for provider in PROVIDERS:
            options = [True, "true", False, "false"] if isinstance(provider, OpenAI) else ["true", "false"]
            question = Choice("Which?", choices=options)
            body = build_request(provider, {"answer": question}, None, ())
            if isinstance(provider, OpenAI):
                values = [entry["value"] for entry in body["questions"][0]["choices"]]
                self.assertEqual(values, options)
                self.assertIs(values[0], True)
                self.assertIs(values[2], False)
            else:
                self.assertEqual(list(body["questions"]["answer"]["criteria"]), options)


if __name__ == "__main__":
    unittest.main()
