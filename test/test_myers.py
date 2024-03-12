from unittest import TestCase

from graphtage.myers import Edit, myers


class TestMyers(TestCase):
    def test_myers(self):
        result = myers([1, 2, 3, 4], [2, 3, 4, 4, 5])
        self.assertEqual([(Edit.REMOVE, 1), (Edit.KEEP, 2), (Edit.KEEP, 3), (Edit.KEEP, 4), (Edit.INSERT, 4),
                          (Edit.INSERT, 5)], result)
