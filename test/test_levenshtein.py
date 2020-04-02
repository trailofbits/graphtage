import random
from typing import List
from unittest import TestCase

from tqdm import trange

from graphtage.levenshtein import EditDistance, string_edit_distance


def levenshtein_distance(s: str, t: str) -> int:
    """Canonical implementation of the Levenshtein distance metric"""
    rows = len(s) + 1
    cols = len(t) + 1
    dist: List[List[int]] = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i

    col = row = 0
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row - 1][col] + 1,
                                 dist[row][col - 1] + 1,
                                 dist[row - 1][col - 1] + cost)

    return dist[row][col]


LETTERS: str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class TestEditDistance(TestCase):
    def test_string_edit_distance(self):
        for _ in trange(1000):
            str1_len = random.randint(50, 100)
            str2_len = random.randint(50, 100)
            str_from = random.choices(LETTERS, k=str1_len)
            str_to = random.choices(LETTERS, k=str2_len)
            distance: EditDistance = string_edit_distance(str_from, str_to)
            while distance.tighten_bounds():
                pass
            self.assertTrue(distance.bounds().definitive())
            self.assertTrue(distance.bounds().finite)
            self.assertEqual(levenshtein_distance(str_from, str_to), distance.bounds().upper_bound)
