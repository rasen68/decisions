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

    def test_clients_resolve_typed_integrations(self):
        from decisions import client
        from decisions.providers.integrations import resolve

        self.assertIs(client.resolve, resolve)

    def test_provider_exports_and_cli_entry_point(self):
        self.assertEqual(OpenAI().name, "openai")
        self.assertEqual(TypeSafe().name, "typesafe")
        self.assertEqual(Jeff("http://localhost:8765").name, "jeff")
        self.assertTrue(callable(main))
