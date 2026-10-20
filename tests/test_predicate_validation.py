import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from decisions import AsyncClient, Client, InvalidResponseError, PredicateAnswer
from decisions.providers import OpenAI
from test_response_structure import PROVIDERS, QUESTIONS, valid_response


class PredicateValidationTests(unittest.TestCase):
    def call_client(self, provider, raw, asynchronous, method="decide"):
        response = httpx.Response(200)
        argument = QUESTIONS if method == "decide" else "Ready?"
        with patch.object(response, "json", return_value=raw):
            if asynchronous:
                async def call():
                    async with AsyncClient(provider) as client:
                        with patch.object(client, "_send", AsyncMock(return_value=response)) as send:
                            try:
                                return await getattr(client, method)(argument, "private-state")
                            finally:
                                send.assert_awaited_once()
                return asyncio.run(call())
            with Client(provider) as client:
                with patch.object(client, "_send", return_value=response) as send:
                    try:
                        return getattr(client, method)(argument, "private-state")
                    finally:
                        send.assert_called_once()

    def assert_rejected(self, change_answer):
        for provider in PROVIDERS:
            for asynchronous in (False, True):
                for index, name in enumerate(QUESTIONS):
                    with self.subTest(provider=provider.name, asynchronous=asynchronous, answer=name):
                        raw = valid_response(provider)
                        field = "probability" if isinstance(provider, OpenAI) else "noul"
                        answer = raw["answers"][index if isinstance(provider, OpenAI) else name]
                        change_answer(answer, field)
                        with self.assertRaises(InvalidResponseError) as caught:
                            self.call_client(provider, raw, asynchronous)
                        self.assertEqual(str(caught.exception), "")

    def test_probability_field_is_required(self):
        self.assert_rejected(lambda answer, field: answer.pop(field))

    def test_wrong_provider_field_does_not_replace_required_field(self):
        def replace(answer, field):
            answer["noul" if field == "probability" else "probability"] = answer.pop(field)
        self.assert_rejected(replace)


if __name__ == "__main__":
    unittest.main()
