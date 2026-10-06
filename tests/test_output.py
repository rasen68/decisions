import unittest

from decisions.api import DecisionError
from decisions.output import format_decision


class OutputTests(unittest.TestCase):
    def test_readable_answers_in_question_order(self):
        questions = [{"type": "predicate", "instructions": "Damaged?"}, {"type": "score", "name": "severity", "instructions": "Severity?", "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}]}, {"type": "choice", "instructions": "Department?", "choices": [{"value": "billing"}, {"value": "support"}]}, {"type": "predicate", "instructions": "Restricted?"}]
        response = {"answers": [{"type": "predicate", "name": None, "probability": 0.95}, {"type": "score", "name": "severity", "score": 1.1, "confidence": 0.55, "probabilities": [{"value": 0, "label": "low", "probability": 0.1}, {"value": 1, "label": "medium", "probability": 0.7}, {"value": 2, "label": "high", "probability": 0.2}]}, {"type": "choice", "name": None, "choice": "support", "confidence": 0.93, "probabilities": [{"value": "billing", "probability": 0.05}, {"value": "support", "probability": 0.95}]}, {"type": "refusal", "name": None}]}
        text = format_decision(response, questions)
        for expected in ("1. Damaged?", "95.0%", "2. severity", "Score: 1.100 / 2", "Confidence: 55.0%", "medium", "70.0%", "3. Department?", 'Choice: "support"', "Confidence: 93.0%", "4. Restricted?", "Refused"):
            self.assertIn(expected, text)
        self.assertLess(text.index("1. Damaged?"), text.index("2. severity"))
        self.assertLess(text.index("2. severity"), text.index("3. Department?"))

    def test_boolean_choice_is_readable(self):
        question = {"type": "choice", "instructions": "Which?", "choices": [{"value": True}, {"value": "true"}]}
        response = {"answers": [{"type": "choice", "choice": True, "confidence": 0.8, "probabilities": [{"value": True, "probability": 0.8}, {"value": "true", "probability": 0.2}]}]}
        text = format_decision(response, [question])
        self.assertIn("Choice: true", text)
        self.assertIn("    true: 80.0%", text)
        self.assertIn('    "true": 20.0%', text)

    def test_malformed_answers_raise_useful_error(self):
        question = {"type": "predicate", "instructions": "Q"}
        cases = [{}, {"answers": []}, {"answers": [None]}, {"answers": [{"type": "predicate"}]}, {"answers": [{"type": "predicate", "probability": "high"}]}, {"answers": [{"type": "predicate", "probability": float("nan")}]}, {"answers": [{"type": "predicate", "probability": True}]}, {"answers": [{"type": "future"}]}, {"answers": [{"type": "choice", "choice": "a", "confidence": 0.7, "probabilities": []}]}]
        for response in cases:
            with self.subTest(response=response), self.assertRaises(DecisionError):
                format_decision(response, [question])


if __name__ == "__main__":
    unittest.main()
