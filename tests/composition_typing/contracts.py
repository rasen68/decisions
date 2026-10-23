"""Strict internal composition examples. Unused ignores indicate a lost boundary."""

from typing import Any, assert_type

from decisions.providers.composition import (
    Assembler, Decoder, Encoder, HTTPBinding, Integration, ResponsePipeline,
)
from decisions.providers.integrations import (
    ContentInput, OpenAIQuestions, OpenAIValidated, SeparateImageInput,
    SystemOneQuestions, SystemOneValidated, content_input, decode_openai,
    decode_systemone, identity_payload, openai_body, openai_questions,
    separate_image_input, systemone_body, systemone_questions, validate_openai,
)
from decisions.types import Result

questions: Encoder[OpenAIQuestions] = openai_questions
inputs: Encoder[ContentInput] = content_input
assembler: Assembler[OpenAIQuestions, ContentInput, dict[str, Any]] = openai_body
pipeline: ResponsePipeline[Any, OpenAIValidated, Result] = ResponsePipeline(
    identity_payload, validate_openai, decode_openai,
)
integration: Integration[OpenAIQuestions, ContentInput, dict[str, Any], Any, OpenAIValidated, Result] = Integration(
    (), questions, inputs, assembler, HTTPBinding(), pipeline,
)
assert_type(integration.questions, Encoder[OpenAIQuestions])

wrong_questions: Encoder[OpenAIQuestions] = systemone_questions  # type: ignore[assignment]
wrong_inputs: Encoder[ContentInput] = separate_image_input  # type: ignore[assignment]
wrong_assembler: Assembler[SystemOneQuestions, SeparateImageInput, dict[str, Any]] = openai_body  # type: ignore[assignment]
wrong_decoder: Decoder[OpenAIValidated, Result] = decode_systemone  # type: ignore[assignment]
wrong_systemone_decoder: Decoder[SystemOneValidated, Result] = decode_openai  # type: ignore[assignment]
wrong_registration: Integration[OpenAIQuestions, ContentInput, dict[str, Any], Any, OpenAIValidated, Result] = Integration(
    (), questions, inputs, systemone_body, HTTPBinding(), pipeline,  # type: ignore[arg-type]
)
wrong_pipeline: ResponsePipeline[Any, OpenAIValidated, Result] = ResponsePipeline(
    identity_payload, validate_openai, decode_systemone,  # type: ignore[arg-type]
)
