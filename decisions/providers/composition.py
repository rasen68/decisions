"""Internal typed components for preparing decision requests and decoding responses."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..types import Image, Question, State
from ..types.inputs import validate_images, validate_state
from ..types.questions import validate_questions

AUTH = {"authorization", "proxy-authorization"}

# Generic typevars for values passed between integration stages
Q = TypeVar("Q") #uestion
I = TypeVar("I") #nput
B = TypeVar("B") #ody
P = TypeVar("P") #ayload
V = TypeVar("V") #alidated payload
R = TypeVar("R") #esult
E = TypeVar("E") #ncoded value


@dataclass(frozen=True)
class PublicRequest:
    """Unencoded questions, state and images"""
    questions: Mapping[str, Question]
    state: State | None
    images: Sequence[Image]


@dataclass(frozen=True)
class Target:
    """Basic HTTP info including private auth"""
    provider: str
    model: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Header:
    """A component's request-specific header requirement such as image opt-in"""
    name: str
    value: str = field(repr=False)


@dataclass(frozen=True)
class Encoded(Generic[E]):
    """Typed question or input encoding plus any headers required to send it"""
    value: E
    headers: tuple[Header, ...] = ()


@dataclass(frozen=True)
class PreparedRequest(Generic[B]):
    """Method, URL, merged headers and body ready for either client to send"""
    method: str
    url: str
    headers: Mapping[str, str] = field(repr=False)
    body: B


def merge_headers(*groups: tuple[Header, ...]) -> dict[str, str]:
    """Components may agree on a header but can't override auth or each other"""
    merged: dict[str, str] = {}
    names: dict[str, str] = {}
    for group in groups:
        for header in group:
            key = header.name.lower()
            if key in AUTH:
                raise ValueError("Auth headers belong to the HTTP binding")
            if key in names:
                if merged[names[key]] != header.value:
                    raise ValueError(f"Conflicting reqs for {header.name}")
            else:
                names[key] = header.name
                merged[header.name] = header.value
    return merged


# Reject unsupported request/target combinations before encoding.
InputRule = Callable[[PublicRequest, Target], None]

# Encode questions or combined state/images without host support checks.
Encoder = Callable[[PublicRequest, Target], Encoded[E]]

# Combine compatible encoded values into a body, including host wrappers.
Assembler = Callable[[Q, I, Target], B]

# Attach HTTP details to an assembled body without sending it.
Binding = Callable[[B, tuple[Header, ...], Target], PreparedRequest[B]]


@dataclass(frozen=True)
class HTTPBinding:
    """Prepare HTTP details from target defaults & component headers w/o I/O"""
    method: str = "POST"

    def __call__(self, body: B, headers: tuple[Header, ...], target: Target) -> PreparedRequest[B]:
        defaults = tuple(Header(n, v) for n, v in target.headers.items()
                         if n.lower() not in AUTH)
        merged = merge_headers(defaults, headers)
        merged.update((n, v) for n, v in target.headers.items()
                      if n.lower() not in AUTH)

        return PreparedRequest(self.method, target.url, merged, body)


# Check a host envelope and extract its payload from untrusted HTTP JSON.
Unwrapper = Callable[[Any], P]

# Check payload structure and answer completeness before conversion.
Validator = Callable[[PublicRequest, P], V]

# Convert a validated payload, retaining the original JSON for Result.raw.
Decoder = Callable[[PublicRequest, Target, V, Any], R]


@dataclass(frozen=True)
class ResponsePipeline(Generic[P, V, R]):
    """Unwrap, validate, & convert a response while retaining raw JSON"""

    unwrap: Unwrapper[P]
    validate: Validator[P, V]
    convert: Decoder[V, R]

    def decode(self, request: PublicRequest, target: Target, raw: Any) -> R:
        payload = self.unwrap(raw)
        validated = self.validate(request, payload)
        return self.convert(request, target, validated, raw)


@dataclass(frozen=True)
class Integration(Generic[Q, I, B, P, V, R]):
    """Composition of all of the above for a host/model"""

    rules: tuple[InputRule, ...]
    questions: Encoder[Q]
    inputs: Encoder[I]
    assemble: Assembler[Q, I, B]
    bind: Binding[B]
    response: ResponsePipeline[P, V, R]

    def prepare(self, request: PublicRequest, target: Target) -> PreparedRequest[B]:
        validate_questions(request.questions)
        validate_state(request.state)
        validate_images(request.images)
        for rule in self.rules:
            rule(request, target)
        questions = self.questions(request, target)
        inputs = self.inputs(request, target)
        headers = merge_headers(questions.headers, inputs.headers)
        body = self.assemble(questions.value, inputs.value, target)
        return self.bind(body, tuple(Header(name, value) for name, value in headers.items()), target)
