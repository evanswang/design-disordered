from unittest import TestCase

from utils import merge_dicts


class TestMergeDicts(TestCase):
    def test_basic_merge(self):
        original = {"a": 1, "b": 2}
        new = {"b": 3, "c": 4}
        expected = {"a": 1, "b": 3, "c": 4}
        result = merge_dicts(original, new)
        self.assertEqual(result, expected)
