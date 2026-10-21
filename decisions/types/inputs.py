"""State and images shared by the supported providers."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
import mimetypes
from pathlib import Path
from typing import TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]
State: TypeAlias = str | JSONObject | list[JSONValue]


def validate_state(state: State | None) -> None:
    if state is not None and not isinstance(state, (str, dict, list)):
        raise TypeError("State must be a string, JSON object or array, or None")
    _validate_visit(state, set())

def _validate_visit(value: JSONValue, active: set[int]) -> None:
    match value:
        case None | str() | int() | bool():
            return
        case float():
            if not math.isfinite(value):
                raise ValueError("State numbers must be finite")
            return
        case dict():
            if not all(isinstance(key, str) for key in value):
                raise TypeError("JSON object keys must be strings")
            children = value.values()
        case list():
            children = value
        case _:
            raise TypeError("State contains a value that is not a JSON type")

    # Check for cycles, do this after validating primitives as they share ids
    if id(value) in active:
        raise ValueError("State must not contain cycles")
    active.add(id(value))

    # Validate our children
    try:
        for child in children:
            _validate_visit(child, active)
    finally:
        active.remove(id(value))

@dataclass
class Image:
    data: bytes
    mime_type: str

    def __post_init__(self) -> None:
        validate_image(self)

    @classmethod
    def from_path(cls, path: str | Path, *, mime_type: str | None = None) -> "Image":
        data = Path(path).read_bytes()
        if mime_type is None:
            mime_type = mimetypes.guess_type(str(path))[0]
            if mime_type is None:
                raise ValueError("Cannot infer image MIME type from filename")
        return cls(data, mime_type)

    @classmethod
    def from_bytes(cls, data: bytes, *, mime_type: str) -> "Image":
        return cls(data, mime_type)


def validate_image(image: Image) -> None:
    if not isinstance(image, Image):
        raise TypeError("Images must contain Image objects")
    if not isinstance(image.data, bytes):
        raise TypeError("Image data must be bytes")
    if not image.data:
        raise ValueError("Image data must not be empty")
    if not isinstance(image.mime_type, str):
        raise TypeError("Image MIME type must be a string")
    if image.mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError("Unsupported image MIME type")


def validate_images(images: Sequence[Image]) -> None:
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
        raise TypeError("Images must be a sequence of Image objects")
    for image in images:
        validate_image(image)
