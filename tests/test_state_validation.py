from collections import UserDict
from decimal import Decimal
import unittest

from decisions.types import inputs


class StateValidationTests(unittest.TestCase):
    def test_top_level_state_types(self):
        for state in (True, False, 0, 1.5, b"text", ("text",), {"text"}, UserDict({"a": 1}), object()):
            with self.subTest(state=state):
                with self.assertRaises(TypeError):
                    inputs.validate_state(state)

    def test_nested_values_require_json_types(self):
        for value in (b"text", (1, 2), {1, 2}, Decimal("0.5"), object(), UserDict({"a": 1})):
            for state in ([value], {"outer": [value]}):
                with self.subTest(value=value, state=state):
                    with self.assertRaises(TypeError):
                        inputs.validate_state(state)

    def test_object_keys_require_strings_at_every_depth(self):
        for key in (1, 1.5, True, None, b"key", ("key",)):
            for state in ({key: "value"}, [{"outer": {key: "value"}}]):
                with self.subTest(key=key, state=state):
                    with self.assertRaises(TypeError):
                        inputs.validate_state(state)

    def test_nonfinite_numbers_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for state in ([value], {"outer": [value]}):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        inputs.validate_state(state)

    def test_large_finite_numbers_are_not_converted(self):
        integer = 10**400
        state = {"values": [integer, -integer, 1.7976931348623157e308, -0.0]}
        inputs.validate_state(state)
        self.assertIs(state["values"][0], integer)
        self.assertEqual(state["values"][1], -integer)

    def test_cycles_are_rejected(self):
        array = []
        array.append(array)
        obj = {}
        obj["self"] = obj
        mixed = {"items": []}
        mixed["items"].append(mixed)
        left, right = {}, {}
        left["next"], right["next"] = right, left
        for state in (array, obj, mixed, left):
            with self.subTest(container=type(state)):
                with self.assertRaises(ValueError):
                    inputs.validate_state(state)
        self.assertIs(array[0], array)
        self.assertIs(obj["self"], obj)

    def test_shared_containers_are_not_cycles(self):
        shared = {"values": [True, None, 0.5]}
        state = {"left": shared, "right": shared, "array": [shared, shared["values"]]}
        inputs.validate_state(state)
        self.assertIs(state["left"], state["right"])
        self.assertIs(state["array"][1], shared["values"])

    def test_valid_state_is_not_changed(self):
        for state in (None, "", " \n text \t", {}, [], {"": [None, True, False, 0, -2, 1.5, "text", {"nested": []}]}):
            with self.subTest(state=state):
                before = repr(state)
                inputs.validate_state(state)
                self.assertEqual(repr(state), before)


if __name__ == "__main__":
    unittest.main()
