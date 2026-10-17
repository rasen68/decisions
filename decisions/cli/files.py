"""Read question files and shared text or image input."""

import base64
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


def read_image(filename: str) -> dict[str, str]:
    data = Path(filename).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise QuestionError(f"image {filename!r} must be PNG, JPEG, GIF, or WebP")
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}


def read_input(arguments: Any, stdin: TextIO) -> str | list[dict[str, Any]]:
    text = None
    if arguments.text is not None:
        text = arguments.text
    elif arguments.input_file is not None and arguments.input_file != "-":
        text = Path(arguments.input_file).read_text(encoding="utf-8")
    elif arguments.input_file == "-" or not arguments.images:
        if arguments.input_file is None and stdin.isatty():
            raise QuestionError("provide an input file, --text TEXT, --image PATH, or piped stdin; use - to read terminal input")
        text = stdin.read()
    if not arguments.images:
        return text
    parts = []
    if text is not None:
        parts.append({"type": "input_text", "text": text})
    parts.extend(read_image(filename) for filename in arguments.images)
    return [{"role": "user", "content": parts}]
