"""State and images shared by the supported providers."""

from dataclasses import dataclass
import mimetypes
from pathlib import Path
from typing import TypeAlias, cast

JSONValue: TypeAlias = (
    str | int | float | bool | None
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)
JSONObject: TypeAlias = dict[str, JSONValue]
State: TypeAlias = str | JSONObject | list[JSONValue]

@dataclass
class Image:
    data: bytes
    mime_type: str

    @classmethod
    def from_path(cls, path: str | Path, *, mime_type: str | None = None) -> "Image":
        return cls(Path(path).read_bytes(), mime_type if mime_type is not None else cast(str, mimetypes.guess_type(str(path))[0]))

    @classmethod
    def from_bytes(cls, data: bytes, *, mime_type: str) -> "Image":
        return cls(data, mime_type)
