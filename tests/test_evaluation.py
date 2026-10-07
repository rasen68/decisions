import unittest

from decisions.api import DecisionError
from decisions.evaluation import compile_expression
from decisions.questions import QuestionError


class EvaluationTests(unittest.TestCase):
    questions = [
        {"type": "predicate", "name": "damaged", "instructions": "Damage?"},
        {"type": "choice", "name": "department", "instructions": "Where?", "choices": [{"value": "support"}, {"value": True}, {"value": False}, {"value": "true"}]},
        {"type": "score", "instructions": "Severity?", "levels": [{"label": "low"}, {"label": "high"}]},
    ]
    answers = [
        {"type": "predicate", "probability": 0.9},
        {"type": "choice", "choice": "support", "confidence": 0.8, "probabilities": [
            {"value": "support", "probability": 0.7}, {"value": True, "probability": 0.2},
            {"value": False, "probability": 0.05}, {"value": "true", "probability": 0.05}]},
        {"type": "score", "score": 0.5, "confidence": 0.7},
    ]

    def evaluate(self, expression):
        return compile_expression(expression, self.questions)(self.answers)

    def test_predicate_boundary_and_negation(self):
        self.assertTrue(self.evaluate('damaged >= 0.9'))
        self.assertFalse(self.evaluate('not (q1 >= 0.9)'))
        self.assertFalse(self.evaluate('q1 <= 0.1'))

    def test_predicate_shorthand(self):
        self.assertTrue(self.evaluate('damaged >= 0.9'))
        self.assertTrue(self.evaluate('q1 >= .9 and department.confidence >= .8'))
        self.assertFalse(self.evaluate('not (damaged >= .9)'))
        self.assertFalse(self.evaluate('q1 > .9'))
        for expression in ('damaged.probability >= .9', 'q1.probability >= .9', 'missing >= .9', 'q1 == true'):
            with self.subTest(expression=expression), self.assertRaises(QuestionError):
                compile_expression(expression, self.questions)

    def test_boolean_precedence_and_grouping(self):
        self.assertTrue(self.evaluate('q1 < 0.1 and q3 > 0.9 or department.confidence == 0.8'))
        self.assertFalse(self.evaluate('q1 < 0.1 and (q3 > 0.9 or department.confidence == 0.8)'))

    def test_choice_score_and_typed_booleans(self):
        self.assertTrue(self.evaluate('department == "support" and q3 >= .5 and q3.confidence != 0'))
        self.assertFalse(self.evaluate('department == true'))
        self.assertTrue(self.evaluate('department != false'))
        self.assertFalse(self.evaluate('department == "true"'))

    def test_primary_fields_are_bare_and_only_confidence_accepts_a_dot(self):
        self.assertTrue(self.evaluate('q2 == "support" and q3 >= .5'))
        self.assertTrue(self.evaluate('department == "support" and department.confidence >= .8'))
        for expression in ('department.choice == "support"', 'q2.choice == "support"',
                           'q3.score >= .5', 'q1.probability >= .9', 'q1.confidence >= .9',
                           'q2.probability > 0', 'q3.choice == "high"'):
            with self.subTest(expression=expression), self.assertRaises(QuestionError):
                compile_expression(expression, self.questions)

    def test_invalid_expressions_and_types(self):
        for expression in ('', 'true', 'q1', 'missing > .5', 'q1.confidence > .5',
                           'q2 > "support"', 'q3 == true', 'q1 == "0.9"',
                           'q1 > 0 < 1', '__import__("os")', 'q1.__class__ == "x"',
                           'q1 + 1 > 0', 'q1 > 1e999'):
            with self.subTest(expression=expression), self.assertRaises(QuestionError):
                compile_expression(expression, self.questions)

    def test_probability_of_a_non_selected_choice(self):
        self.assertTrue(self.evaluate('department[true] >= .2'))
        self.assertFalse(self.evaluate('department[true] > .2'))
        self.assertTrue(self.evaluate('q2["support"] >= .7 and department.confidence >= .8'))
        self.assertFalse(self.evaluate('department["true"] >= .2'))
        self.assertTrue(self.evaluate('department[false] <= .05'))

    def test_invalid_outcome_references(self):
        for expression in ('department["missing"] > .1', 'department[1] > .1',
                           'q1[true] > .1', 'q3["low"] > .1', 'department["support"] == true',
                           'department["support"].confidence > .1', 'department["support"][0] > .1'):
            with self.subTest(expression=expression), self.assertRaises(QuestionError):
                compile_expression(expression, self.questions)

    def test_outcome_probability_errors_even_in_unused_branches(self):
        evaluate = compile_expression('q1 > .1 or department[true] > .1', self.questions)
        for probabilities in ([{"value": "support", "probability": .7}],
                              [{"value": True, "probability": .2}, {"value": True, "probability": .2}],
                              [{"value": True, "probability": 2}], [{"value": True, "probability": True}]):
            answers = [dict(answer) for answer in self.answers]
            answers[1]["probabilities"] = probabilities
            with self.subTest(probabilities=probabilities), self.assertRaises(DecisionError):
                evaluate(answers)

    def test_alias_collision_is_rejected_only_when_used(self):
        questions = [{"type": "predicate", "name": "q2"}, {"type": "predicate"}]
        compile_expression('q1 > .5', questions)
        with self.assertRaises(QuestionError):
            compile_expression('q2 > .5', questions)
