import json
import unittest
from unittest.mock import patch

import httpx

from decisions import AsyncClient, Choice, ChoiceAnswer, Client, Image, Predicate, PredicateAnswer, ScoreAnswer
from decisions.providers import Jeff, OpenAI, TypeSafe


def response_for(request):
    body = json.loads(request.content)
    questions = body["questions"]
    openai = isinstance(questions, list)
    answers = [] if openai else {}
    items = [(question["name"], question) for question in questions] if openai else questions.items()
    for name, question in items:
        kind = question["type"]
        if kind in ("predicate", "noul"):
            answer = {"type": kind, "probability" if openai else "noul": .9}
        elif kind == "choice":
            keys = [entry["value"] for entry in question["choices"]] if openai else list(question["criteria"])
            probabilities = [{"value": key, "probability": probability} for key, probability in zip(keys, [.8, .2])] if openai else dict(zip(keys, [.8, .2]))
            answer = {"type": kind, "choice": keys[0], "confidence": .123, "probabilities": probabilities}
        else:
            probabilities = [{"value": index, "label": level["label"], "probability": probability} for index, (level, probability) in enumerate(zip(question["levels"], [.2, .8]))] if openai else {"0": .2, "1": .8}
            answer = {"type": kind, "score": .8, "confidence": .456, "probabilities": probabilities}
        if openai:
            answers.append(answer)
        else:
            answers[name] = answer
    return httpx.Response(200, json={"model": "reported", "answers": answers, "usage": {"input_tokens": 12, "output_tokens": 3}})


class ClientTests(unittest.TestCase):
    def test_default_provider_credentials_and_close(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-env"}), Client() as client:
            self.assertEqual(client.provider, OpenAI())
            self.assertEqual(client._http.headers["Authorization"], "Bearer synthetic-env")
            self.assertEqual(client._http.timeout.read, 60)
        self.assertTrue(client._http.is_closed)
        client.close()

    def test_provider_credentials_and_endpoints(self):
        for provider, authorization, url in (
            (OpenAI(api_key="synthetic-explicit", base_url="https://openai.test/v1/"), "Bearer synthetic-explicit", "https://openai.test/v1/decisions"),
            (TypeSafe(), "Bearer synthetic-typesafe", "https://api.typesafe.ai/v1/systemone"),
            (Jeff("http://localhost:8765/"), None, "http://localhost:8765/v1/systemone"),
            (Jeff("http://localhost:8765", api_key="synthetic-jeff"), "Bearer synthetic-jeff", "http://localhost:8765/v1/systemone"),
        ):
            requests = []
            def handler(request):
                requests.append(request)
                return response_for(request)
            factory = httpx.Client
            with self.subTest(provider=provider), patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-openai", "TYPESAFE_API_KEY": "synthetic-typesafe"}), patch("decisions.client.httpx.Client", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
                client = Client(provider, timeout=23)
                self.assertEqual(client.predicate("Ready?", "Text"), PredicateAnswer(.9))
                self.assertEqual(str(requests[0].url), url)
                self.assertEqual(requests[0].headers.get("Authorization"), authorization)
                self.assertEqual(requests[0].extensions["timeout"]["read"], 23)
                client.close()
                self.assertTrue(client._http.is_closed)

    def test_convenience_calls_and_multiquestion_use_one_request_each(self):
        for provider in (OpenAI(api_key="synthetic"), TypeSafe(api_key="synthetic"), Jeff("http://localhost:8765")):
            requests = []
            def handler(request):
                requests.append(request)
                return response_for(request)
            factory = httpx.Client
            with self.subTest(provider=provider), patch("decisions.client.httpx.Client", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)), Client(provider) as client:
                self.assertIs(Client.noul, Client.predicate)
                self.assertEqual(client.noul(question="Ready?", state="Text"), PredicateAnswer(.9))
                choices = [True, "true"] if isinstance(provider, OpenAI) else ["support", "billing"]
                self.assertEqual(client.choice("Team?", "Text", choices=choices), ChoiceAnswer(choices[0], .123, dict(zip(choices, [.8, .2]))))
                self.assertEqual(client.score("Urgent?", state="Text", levels={"low": "Later", "high": "Now"}), ScoreAnswer(.8, .456, {"low": .2, "high": .8}))
                result = client.decide(questions={"ready": Predicate("Ready?"), "team": Choice("Team?", choices=["support", "billing"])}, state="Text")
                self.assertEqual(list(result.answers), ["ready", "team"])
                self.assertEqual(result.model, "reported")
                self.assertEqual(len(requests), 4)
                if not isinstance(provider, TypeSafe):
                    self.assertEqual(client.predicate("Visible?", images=[Image(b"a", "image/png")]), PredicateAnswer(.9))


class AsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_convenience_calls_multiquestion_and_lifecycle(self):
        self.assertIs(AsyncClient.noul, AsyncClient.predicate)
        for provider in (OpenAI(api_key="synthetic"), TypeSafe(api_key="synthetic"), Jeff("http://localhost:8765")):
            requests = []
            async def handler(request):
                requests.append(request)
                return response_for(request)
            factory = httpx.AsyncClient
            with self.subTest(provider=provider), patch("decisions.client.httpx.AsyncClient", side_effect=lambda **kwargs: factory(transport=httpx.MockTransport(handler), **kwargs)):
                async with AsyncClient(provider) as client:
                    self.assertEqual(await client.noul("Ready?", "Text"), PredicateAnswer(.9))
                    self.assertEqual(await client.predicate(question="Ready?", state="Text"), PredicateAnswer(.9))
                    choices = [False, "false"] if isinstance(provider, OpenAI) else ["support", "billing"]
                    self.assertEqual(await client.choice("Team?", state="Text", choices=choices), ChoiceAnswer(choices[0], .123, dict(zip(choices, [.8, .2]))))
                    self.assertEqual(await client.score("Urgent?", "Text", levels=["low", "high"]), ScoreAnswer(.8, .456, {"low": .2, "high": .8}))
                    result = await client.decide({"ready": Predicate("Ready?")}, "Text")
                    self.assertEqual(result["ready"].probability, .9)
                    self.assertEqual(len(requests), 5)
                    if not isinstance(provider, TypeSafe):
                        self.assertEqual(await client.predicate("Visible?", images=[Image(b"a", "image/png")]), PredicateAnswer(.9))
                self.assertTrue(client._http.is_closed)
                await client.aclose()

    async def test_explicit_async_close_and_default_credentials(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic"}):
            client = AsyncClient()
        self.assertEqual(client._http.headers["Authorization"], "Bearer synthetic")
        await client.aclose()
        self.assertTrue(client._http.is_closed)
