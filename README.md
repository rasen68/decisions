# decisions

A Python library for asking classification and scoring questions through [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions), [TypeSafe Jev](https://docs.typesafe.ai/), or a [Jeff](https://github.com/firelex/jeff) server. It provides a shared interface, typed answers, and synchronous and asynchronous clients, using HTTPX rather than depending on each provider's SDK.

Ask whether a statement is true, choose between options, or score input against ordered levels. Send one question at a time or several named questions in a single request.

The package also includes an OpenAI-only command-line tool. See the [CLI README](decisions/cli/README.md) for command-line installation, usage, graphical output, boolean gates, and conditional shell commands.

## Installation

Requires Python 3.11 or newer. From the repository root:

```sh
pip install .
```

HTTPX is the only direct runtime dependency. Provider access requires the appropriate API credentials or a running Jeff server.

## Quick start

OpenAI is the default provider. Set `OPENAI_API_KEY`, then create a reusable client.

We support single-question functions that return only the core result (probabilities, choice/score, confidence):

```python
from decisions import Client

with Client() as client:
    damaged = client.predicate(
        "Is the product damaged?",
        "The screen arrived broken.",
    )
    print(damaged.probability)

    department = client.choice(
        "Which department should handle this?",
        "I was charged twice.",
        choices={"billing": "Payments and refunds", "support": "Technical problems"},
        # Or: just choices=["billing", "support"]
    )
    print(department.choice, department.confidence)
    print(department.probabilities)

    urgency = client.score(
        "How urgent is this?",
        "The app won't start.",
        levels=["low", "medium", "high"],
    )
    print(urgency.score, urgency.confidence)
    print(urgency.probabilities)
```

You can also use reusable question objects with `decide` to ask several questions about the same input in one request:

```python
from decisions import Choice, Client, Predicate, Score

questions = {
    "damaged": Predicate("Is the product damaged?"),
    "department": Choice("Which department?", choices=["billing", "support"]),
    "urgency": Score("How urgent?", levels=["low", "medium", "high"]),
}

with Client() as client:
    result = client.decide(questions, "The screen arrived broken.")

print(result["damaged"].probability)
print(result["department"].choice)
print(result["urgency"].score)
print(result.provider, result.model)
print(result.usage.input_tokens, result.usage.output_tokens)
```

Answers and probability mappings follow the request's question and option order. `Result.raw` retains the decoded provider response, including provider-specific metadata. The single-question methods return just the answer, without response metadata.

## Providers and supported inputs

Pass a provider configuration to the client:

```python
from decisions import Client
from decisions.providers import Jeff, OpenAI, TypeSafe

# Select one configuration for a client.
provider = TypeSafe(model="jev-latest")
# provider = OpenAI(model="gpt-6-luna")
# provider = Jeff("http://localhost:8765", model="jeff-latest")

with Client(provider) as client:
    answer = client.predicate("Is this about billing?", "I was charged twice.")
```

For OpenAI and TypeSafe providers, set environment variables `OPENAI_API_KEY`/`TYPESAFE_API_KEY` respectively, or pass an api key as `api_key=` to the constructor.

## Images

OpenAI Decisions and Jeff support image inputs, with or without text:

```python
from decisions import Client, Image

with Client() as client:
    answer = client.predicate(
        "Is the product damaged?",
        images=[Image.from_path("photo.jpg")],
    )
    print(answer.probability)
```

`Image.from_path` infers MIME type from the filename, with an optional `mime_type=` override. Unknown inference raises `ValueError`. For bytes, use `Image.from_bytes(data, mime_type="image/png")`.

## Async usage

`AsyncClient` has the same methods and input types as `Client`:

```python
import asyncio
from decisions import AsyncClient

async def main():
    async with AsyncClient() as client:
        answer = await client.predicate("Is this about billing?", "Refund please.")
        print(answer.probability)

asyncio.run(main())
```

Reuse clients to reuse their HTTP connection pools. Context managers close them automatically; otherwise call `client.close()` or `await client.aclose()`. Both clients default to a 60-second timeout, configurable with `timeout=` when constructing them.

## Invalid responses and refusals

Both clients raise `decisions.InvalidResponseError` for malformed JSON or invalid response fields. Answers must match the requested question types. Probabilities and confidence must be finite numbers between zero and one; scores must be within the requested scale. Probability distributions must cover every option, but their sum needn't equal exactly one. Values aren't rounded or normalized.

OpenAI can refuse individual questions. Such answers return `decisions.RefusalAnswer` rather than raising an exception, including from single-question methods and `noul`. Other answers in the same result remain available:

```python
if isinstance(answer, decisions.RefusalAnswer):
    print("The model declined to answer.")
else:
    print(answer.probability)
```

`RefusalAnswer.explanation` is currently `None` because OpenAI's documented decision refusal contains no explanation. The original payload remains in `Result.raw`. TypeSafe and Jeff don't document refusal variants; those variants raise `decisions.InvalidResponseError`.

## Development

Run the tests (synthetic credentials & mocked HTTP) from the repo root:

```sh
python3 -B -m unittest discover -s tests -v
```

Check typings from a consumer environment:

```sh
python3 -m venv .venv/typing-check
.venv/typing-check/bin/python -m pip install '.[typing]'
.venv/typing-check/bin/python scripts/check_typing.py
```

## License

MIT
