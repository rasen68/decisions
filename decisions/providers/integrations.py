"""Registrations for existing integrations, without new model support policies."""

from dataclasses import dataclass
from typing import Any

from .composition import Encoded, HTTPBinding, Integration, PublicRequest, ResponsePipeline, Target
from .config import Jeff, OpenAI, Provider, TypeSafe
from .requests import build_openai_input, build_openai_question, build_systemone_question, data_url
from .responses import _validate_response, convert_result
from ..types import Choice, Result, State


@dataclass(frozen=True)
class OpenAIQuestions:
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class SystemOneQuestions:
    items: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ContentInput:
    content: str | list[dict[str, Any]] | None


@dataclass(frozen=True)
class SeparateImageInput:
    state: State
    images: tuple[str, ...]


@dataclass(frozen=True)
class OpenAIValidated:
    payload: dict[str, Any]


@dataclass(frozen=True)
class SystemOneValidated:
    payload: dict[str, Any]


def text_state(request: PublicRequest, target: Target) -> None:
    if request.state is not None and not isinstance(request.state, str):
        raise ValueError("OpenAI state must be a string or None")


def string_choices(request: PublicRequest, target: Target) -> None:
    for question in request.questions.values():
        if isinstance(question, Choice) and any(isinstance(value, bool) for value in question.choices):
            raise ValueError("This provider does not support boolean choices")


def no_images(request: PublicRequest, target: Target) -> None:
    if request.images:
        raise ValueError("TypeSafe adapter does not support images")


def no_gif(request: PublicRequest, target: Target) -> None:
    if any(image.mime_type == "image/gif" for image in request.images):
        raise ValueError("Jeff does not support GIF images")


def openai_questions(request: PublicRequest, target: Target) -> Encoded[OpenAIQuestions]:
    return Encoded(OpenAIQuestions([build_openai_question(name, question)
                                    for name, question in request.questions.items()]))


def systemone_questions(request: PublicRequest, target: Target) -> Encoded[SystemOneQuestions]:
    return Encoded(SystemOneQuestions({name: build_systemone_question(question)
                                      for name, question in request.questions.items()}))


def content_input(request: PublicRequest, target: Target) -> Encoded[ContentInput]:
    return Encoded(ContentInput(build_openai_input(request.state, request.images)))


def separate_image_input(request: PublicRequest, target: Target) -> Encoded[SeparateImageInput]:
    return Encoded(SeparateImageInput("" if request.state is None else request.state,
                                     tuple(data_url(image) for image in request.images)))


def openai_body(questions: OpenAIQuestions, inputs: ContentInput, target: Target) -> dict[str, Any]:
    return {"model": target.model, "questions": questions.items, "input": inputs.content}


def systemone_body(questions: SystemOneQuestions, inputs: SeparateImageInput, target: Target) -> dict[str, Any]:
    body = {"model": target.model, "questions": questions.items, "state": inputs.state}
    if inputs.images:
        body["images"] = list(inputs.images)
    return body


def identity_payload(raw: Any) -> Any:
    return raw


def validate_openai(request: PublicRequest, payload: Any) -> OpenAIValidated:
    _validate_response(request.questions, payload, openai=True)
    return OpenAIValidated(payload)


def validate_systemone(request: PublicRequest, payload: Any) -> SystemOneValidated:
    _validate_response(request.questions, payload, openai=False)
    return SystemOneValidated(payload)


def decode_openai(request: PublicRequest, target: Target, validated: OpenAIValidated, raw: Any) -> Result:
    return convert_result(target.provider, request.questions, validated.payload, raw, openai=True)


def decode_systemone(request: PublicRequest, target: Target, validated: SystemOneValidated, raw: Any) -> Result:
    return convert_result(target.provider, request.questions, validated.payload, raw, openai=False)


OPENAI: Integration[OpenAIQuestions, ContentInput, dict[str, Any], Any, OpenAIValidated, Result] = Integration(
    rules=(text_state,),
    questions=openai_questions,
    inputs=content_input,
    assemble=openai_body,
    bind=HTTPBinding(),
    response=ResponsePipeline(
        unwrap=identity_payload,
        validate=validate_openai,
        convert=decode_openai,
    ),
)
TYPESAFE: Integration[SystemOneQuestions, SeparateImageInput, dict[str, Any], Any, SystemOneValidated, Result] = Integration(
    rules=(no_images, string_choices),
    questions=systemone_questions,
    inputs=separate_image_input,
    assemble=systemone_body,
    bind=HTTPBinding(),
    response=ResponsePipeline(
        unwrap=identity_payload,
        validate=validate_systemone,
        convert=decode_systemone,
    ),
)
JEFF: Integration[SystemOneQuestions, SeparateImageInput, dict[str, Any], Any, SystemOneValidated, Result] = Integration(
    rules=(no_gif, string_choices),
    questions=systemone_questions,
    inputs=separate_image_input,
    assemble=systemone_body,
    bind=HTTPBinding(),
    response=ResponsePipeline(
        unwrap=identity_payload,
        validate=validate_systemone,
        convert=decode_systemone,
    ),
)

def resolve(provider: Provider) -> (
    Integration[OpenAIQuestions, ContentInput, dict[str, Any], Any, OpenAIValidated, Result]
    | Integration[SystemOneQuestions, SeparateImageInput, dict[str, Any], Any, SystemOneValidated, Result]
):
    match provider:
        case OpenAI():
            return OPENAI
        case TypeSafe():
            return TYPESAFE
        case Jeff():
            return JEFF
    raise TypeError("Unknown provider configuration")
