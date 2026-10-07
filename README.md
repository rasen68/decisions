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

Note that the Decisions API requires that questions have names; if given from the command line, we just assign sequential names q1, q2, ...

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

## Boolean gates

Use `--eval EXPR` to turn answers into an exit status based on a boolean expression. If given, instead of normal exit statuses, it returns 0 when the expression is true, 1 when false, and 2 for refusals and usage, configuration, file, API, or response errors.

```sh
decisions -p "Does the customer report damage?" -i "The screen arrived broken." \
  --eval 'q1 >= 0.9' --quiet

decisions examples/questions.toml input.txt --eval \
  'damaged >= 0.9 and (severity >= 1.5 or department == "support" or department["billing"] >= 0.5)'
```

Reference questions by their names from a question file or by their position as `q1`, `q2`, and so on. Names used in expressions must be identifiers such as `damaged` or `needs_review`; use positional references for other names. Reference the probability of a predicate, the interpolated value of a score, or the maximum of a choice by its name. You can reference the confidence of a choice or score via `name.confidence`. You can reference the individual probability of an option of a choice via `name["option"]`.

We support a Python-like expression grammar with `()`, `and/or/not`, `==/!=`, and `</<=/>/>=` for numeric fields. `not` binds before `and` which binds before `or`. Strings must be quoted.

Answers still print by default, including when the expression is false. `--raw` and `--graphic` work with `--eval`. Use `-q/--quiet` to suppress answers; diagnostics still go to stderr. `--quiet` requires `--eval` and is mutually exclusive with `--raw` and `--graphic`.

## Development

```sh
python3 -B -m unittest discover -s tests -v
```

Tests use synthetic keys and mocked HTTP. They make no live API calls.

## AI use

Code was primarily generated with AI. Documentation was primarily written by me, a human.

## License

MIT
