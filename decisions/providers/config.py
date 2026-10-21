"""Configuration for the supported HTTP providers."""

from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class OpenAI:
    api_key: str | None = field(default=None, repr=False)
    model: str = "gpt-6-luna"
    base_url: str = "https://api.openai.com/v1"
    name: ClassVar[str] = "openai"
    path: ClassVar[str] = "/decisions"
    key_env: ClassVar[str] = "OPENAI_API_KEY"

@dataclass
class TypeSafe:
    api_key: str | None = field(default=None, repr=False)
    model: str = "jev-latest"
    base_url: str = "https://api.typesafe.ai/v1"
    name: ClassVar[str] = "typesafe"
    path: ClassVar[str] = "/systemone"
    key_env: ClassVar[str] = "TYPESAFE_API_KEY"

@dataclass(kw_only=True)
class Jeff:
    base_url: str = field(kw_only=False)
    model: str = "jeff-latest"
    api_key: str | None = field(default=None, repr=False)
    name: ClassVar[str] = "jeff"
    path: ClassVar[str] = "/v1/systemone"
    key_env: ClassVar[None] = None

Provider = OpenAI | TypeSafe | Jeff
