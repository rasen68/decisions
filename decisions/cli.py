"""Command-line parsing and entry point."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from . import __version__
from .api import DecisionError, request_decision
from .evaluation import compile_expression
from .files import load_questions, read_input
from .output import format_decision
from .questions import QuestionError, flag_question, validate_questions


@dataclass
class Arguments:
    questions: list[dict[str, Any]] | None
    question_file: str | None
    input_file: str | None
    input: str | None
    model: str
    raw: bool
    graphic: bool
    expression: str | None
    quiet: bool


class QuestionAction(argparse.Action):
    def __init__(self, *args, question_type: str, **kwargs):
        self.question_type = question_type
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            question = flag_question(self.question_type, values)
        except QuestionError as error:
            parser.error(str(error))
        questions = getattr(namespace, self.dest, None)
        if questions is None:
            questions = []
            setattr(namespace, self.dest, questions)
        questions.append(question)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decisions",
        description="Evaluate questions about shared text with OpenAI Decisions.",
        epilog="With question flags, FILE is the input file. Otherwise, use QUESTIONS.toml/json [INPUT]. "
        "Put input before the flags, or after -- when using --score or --choice. "
        "If no input file or --input is given, read stdin. Set OPENAI_API_KEY for authentication.",
        allow_abbrev=False,
    )
    parser.add_argument("files", nargs="*", metavar="FILE", help="question file and/or input file; - reads input from stdin")
    parser.add_argument("-p", "--predicate", dest="questions", action=QuestionAction, question_type="predicate", metavar="QUESTION", help="estimate the probability that a condition is true; repeatable")
    parser.add_argument("-s", "--score", dest="questions", action=QuestionAction, question_type="score", nargs="+", metavar="ARG", help="QUESTION followed by ordered levels, lowest to highest; repeatable")
    parser.add_argument("-c", "--choice", dest="questions", action=QuestionAction, question_type="choice", nargs="+", metavar="ARG", help="QUESTION followed by choices; repeatable; options accept NAME: DESCRIPTION")
    parser.add_argument("-i", "--input", metavar="TEXT", help="shared input as one argument")
    parser.add_argument("-m", "--model", default="gpt-6-luna", help="model to use (default: %(default)s)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("-r", "--raw", action="store_true", help="print the complete original JSON response")
    output.add_argument("-g", "--graphic", action="store_true", help="print probability bars under each question")
    output.add_argument("-q", "--quiet", action="store_true", help="with --eval, suppress answers and use only the exit status")
    parser.add_argument("--eval", dest="expression", metavar="EXPR", help="evaluate comparisons joined by and/or/not; exit 0 if true, 1 if false, 2 on error")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def parse_arguments(argv: list[str] | None = None) -> Arguments:
    parser = build_parser()
    namespace = parser.parse_intermixed_args(argv)
    question_file = None
    input_file = None
    if namespace.questions is not None:
        if len(namespace.files) > 1:
            parser.error("question files and question flags are mutually exclusive; with flags, provide only one input file")
        try:
            questions = validate_questions(namespace.questions)
        except QuestionError as error:
            parser.error(str(error))
        if namespace.files:
            input_file = namespace.files[0]
    else:
        questions = None
        if not namespace.files:
            parser.error("provide question flags or a TOML/JSON question file")
        if len(namespace.files) > 2:
            parser.error("expected a question file followed by at most one input file")
        question_file = namespace.files[0]
        if len(namespace.files) == 2:
            input_file = namespace.files[1]
    if input_file is not None and namespace.input is not None:
        parser.error("an input file and --input are mutually exclusive")
    if not namespace.model.strip():
        parser.error("--model must not be empty")
    if namespace.quiet and namespace.expression is None:
        parser.error("--quiet requires --eval")
    return Arguments(questions, question_file, input_file, namespace.input, namespace.model, namespace.raw, namespace.graphic, namespace.expression, namespace.quiet)


def silence_broken_pipe() -> None:
    # Redirect the descriptor so Python's final stdout flush cannot raise again.
    try:
        descriptor = sys.stdout.fileno()
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), descriptor)
    except (OSError, ValueError, AttributeError):
        pass


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    result = 0
    try:
        questions = arguments.questions
        if questions is None:
            questions = load_questions(arguments.question_file)
        evaluate = compile_expression(arguments.expression, questions) if arguments.expression is not None else None
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise QuestionError("set OPENAI_API_KEY before making a request")
        input_text = read_input(arguments, sys.stdin)
        raw = request_decision(input_text, questions, arguments.model, api_key)
        if evaluate is not None:
            response = json.loads(raw)
            readable = format_decision(response, questions, graphic=arguments.graphic)
            for index, (question, answer) in enumerate(zip(questions, response["answers"]), 1):
                if answer["type"] == "refusal":
                    raise DecisionError(f"question {index} was refused; cannot evaluate --eval")
                if question["type"] == "choice" and not any(
                    type(answer["choice"]) is type(entry["value"]) and answer["choice"] == entry["value"]
                    for entry in question["choices"]
                ):
                    raise DecisionError(f"invalid OpenAI answer {index}: choice is outside the supplied choices")
            result = 0 if evaluate(response["answers"]) else 1
            if arguments.quiet:
                return result
            output = raw if arguments.raw else readable
        else:
            output = raw if arguments.raw else format_decision(json.loads(raw), questions, graphic=arguments.graphic)
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    except QuestionError as error:
        build_parser().error(str(error))
    except BrokenPipeError:
        silence_broken_pipe()
    except (DecisionError, OSError, UnicodeError) as error:
        print(f"decisions: {error}", file=sys.stderr)
        return 2 if arguments.expression is not None else 1
    except KeyboardInterrupt:
        print("decisions: interrupted", file=sys.stderr)
        return 130
    return result
