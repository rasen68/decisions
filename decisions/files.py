"""Read question files and shared text input."""

import json
from pathlib import Path
import tomllib
from typing import Any, TextIO

from .questions import QuestionError, validate_questions


def load_questions(filename: str) -> list[dict[str, Any]]:
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".toml"):
        raise QuestionError("question files must have a .json or .toml extension")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if suffix == ".json" else tomllib.loads(text)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise QuestionError(f"invalid question file {filename!r}: {error}") from error
    if not isinstance(data, dict) or data.keys() != {"questions"}:
        raise QuestionError("a question file must contain an object with only a questions array")
    return validate_questions(data["questions"])


def read_input(arguments: Any, stdin: TextIO) -> str:
    if arguments.input is not None:
        return arguments.input
    if arguments.input_file is not None and arguments.input_file != "-":
        return Path(arguments.input_file).read_text(encoding="utf-8")
    if arguments.input_file is None and stdin.isatty():
        raise QuestionError("provide an input file, --input TEXT, or piped stdin; use - to read terminal input")
    return stdin.read()
