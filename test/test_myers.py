from unittest import TestCase

from graphtage.myers import Edit, myers


class TestMyers(TestCase):
    def test_myers(self):
        result = myers([1, 2, 3, 4], [2, 3, 4, 4, 5])
        self.assertEqual([(Edit.REMOVE, 1), (Edit.KEEP, 2), (Edit.KEEP, 3), (Edit.KEEP, 4), (Edit.INSERT, 4),
                          (Edit.INSERT, 5)], result)

    def test_custom_comparator(self):
        def cmp(a: str, b: str):
            a_suffix = a[-3:]
            b_suffix = b[-3:]
            return a_suffix == b_suffix and len(a_suffix) >= 3

        result = myers(["abcdefg", "hijklmnop", "qrstuvwxyz"], ["defg", "op", "wxyz"], is_eq=cmp)
        self.assertEqual([
            (Edit.KEEP, 'abcdefg'), (Edit.REMOVE, 'hijklmnop'), (Edit.INSERT, 'op'), (Edit.KEEP, 'qrstuvwxyz')
        ], result)
