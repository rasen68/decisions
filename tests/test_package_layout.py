import unittest

import decisions
from decisions.cli import main
from decisions.types import Choice, Image, Noul, Predicate, Result, Score, Usage
from decisions.providers import Jeff, OpenAI, TypeSafe


class PackageLayoutTests(unittest.TestCase):
    def test_public_exports_reference_the_model_classes(self):
        for model in (Choice, Image, Predicate, Result, Score, Usage):
            self.assertIs(getattr(decisions, model.__name__), model)
        self.assertIs(Noul, Predicate)
        self.assertIs(decisions.Noul, Predicate)

    def test_clients_use_separate_request_and_response_modules(self):
        from decisions import client
        from decisions.providers.requests import build_request
        from decisions.providers.responses import decode_result

        self.assertIs(client.build_request, build_request)
        self.assertIs(client.decode_result, decode_result)

    def test_provider_exports_and_cli_entry_point(self):
        self.assertEqual(OpenAI().name, "openai")
        self.assertEqual(TypeSafe().name, "typesafe")
        self.assertEqual(Jeff("http://localhost:8765").name, "jeff")
        self.assertTrue(callable(main))
