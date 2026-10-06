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

Put an input filename before the question flags. If it follows a score or
choice's options, use `--` so it isn't interpreted as another option:

```sh
decisions -c "Which department?" "billing" "support" -- input.txt
```

## Output

Readable output shows each question and its answer. For example:

```text
1. damaged: Does the customer report a damaged item?
  Probability: 95.0%

2. severity: How severe is the issue?
  Score: 1.100 / 2
  Confidence: 55.0%
    0 (low): 10.0%
    1 (medium): 70.0%
    2 (high): 20.0%

3. department: Which department should handle this?
  Choice: "support"
  Confidence: 93.0%
    "billing": 5.0%
    "support": 95.0%
```

Predicate probability estimates whether the condition is true. Score indices
start at zero; the returned score is a probability-weighted average and may
fall between levels. Choice and score confidence is a separate API field.
String choices are quoted so they remain distinct from boolean choices.
Refusals appear as `Refused` for the affected question.

Use `-r/--raw` to print the complete response JSON, including model, answers,
and usage. Its original spacing and key order are preserved, with a final
newline added if needed:

```sh
decisions -r -p "Does this request a refund?" -i "Refund my order." | jq .
```

Answers go to stdout; diagnostics go to stderr. Exit codes are 0 for success,
including refusals, 1 for file or API failures, 2 for invalid usage or
configuration, and 130 for an interruption. A closed output pipe exits cleanly.
Requests time out after 60 seconds and are not automatically retried.

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
