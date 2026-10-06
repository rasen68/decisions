# decisions

CLI to query [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions). Python >= 3.11.

## Install

From this directory:

```sh
uv tool install .
# Or:
pipx install .
```

Set your API key as environment variable `OPENAI_API_KEY`. The Decisions API is in beta and only supports `gpt-6-luna` right now, but you can set `-m/--model` to anything else too (it won't work yet).

## Basic usage

Supply questions from either a `.json` or `.toml` file following the formats in `examples/`, or via flags (see below). Input will come from either a plaintext file, stdin, or a single argument after `-i/--input`.

```sh
decisions examples/questions.toml input.txt
decisions examples/questions.json -i "The screen arrived broken."
```

## Questions flags

Predicates take one question. Scores and choices take a question followed by one or more options. Repeat and mix the flags to ask multiple questions in one request. Answers will always appear in command order.

```sh
decisions -p "Does the customer report damage?" -i "The screen arrived broken."

decisions input.txt \
  -p "Does the customer report damage?" \
  -s "How severe is the issue?" "low" "medium" "high" \
  -c "Which department should handle this?" "billing" "support"
```

Score levels are ordered from lowest to highest. Option names become their descriptions unless you supply `name: description`. You can mix plain and described options. Flag options are strings; question files also support boolean choice values.

```sh
decisions -s "How severe is the issue?" \
  "low: Cosmetic only" \
  "medium: A workaround exists" \
  "high: The product is unusable" \
  -i "The app fails to open."
```

Put an input filename before the question flags. If it follows a score or choice's options, use `--` so it isn't interpreted as another option:

```sh
decisions -c "Which department?" "billing" "support" -- input.txt
```

## Output

There are three outputs modes: standard, `-r/--raw`, and `-g/--graphic` (pretty).

Standard output example:

```text
1. severity: How severe is the issue?
  Score: 1.100 / 2
  Confidence: 55.0%
    0 (low): 10.0%
    1 (medium): 70.0%
    2 (high): 20.0%
```

Graphic output example:

```text
1. Does the customer report damage?
  Probability: [███████████████████░] 95.0%
```

Raw output example:

```json
{
  "model": "gpt-6-luna",
  "answers": [
    {"type": "predicate", "name": null, "probability": 0.95}
  ],
  "usage": {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 42
  }
}
```

## Development

```sh
python3 -B -m unittest discover -s tests -v
```

Tests use synthetic keys and mocked HTTP. They make no live API calls.

The implementation follows the [Decisions guide]
and [API reference](https://developers.openai.com/api/reference/resources/decisions/methods/create).
The API is currently in public beta and documents `gpt-6-luna` as its supported
model. This CLI handles text input.

## AI use

Code was primarily generated with AI. Documentation was primarily written by me, a human.

## License

MIT
