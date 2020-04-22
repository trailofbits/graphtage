import random
from typing import List
from unittest import TestCase

from tqdm import trange

from graphtage.edits import Edit, Insert, Match, Remove
from graphtage import EditDistance, string_edit_distance


LETTERS: str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class TestEditDistance(TestCase):
    def test_string_edit_distance_reconstruction(self):
        for _ in trange(200):
            str1_len = random.randint(10, 30)
            str2_len = random.randint(10, 30)
            str_from = ''.join(random.choices(LETTERS, k=str1_len))
            str_to = ''.join(random.choices(LETTERS, k=str2_len))
            distance: EditDistance = string_edit_distance(str_from, str_to)
            edits: List[Edit] = list(distance.edits())
            reconstructed_from = ''
            reconstructed_to = ''
            for edit in edits:
                if isinstance(edit, Match):
                    reconstructed_from += edit.from_node.object
                    reconstructed_to += edit.to_node.object
                elif isinstance(edit, Remove):
                    reconstructed_from += edit.from_node.object
                elif isinstance(edit, Insert):
                    reconstructed_to += edit.from_node.object
                else:
                    self.fail()
            self.assertEqual(str_from, reconstructed_from)
            self.assertEqual(str_to, reconstructed_to)

    def test_string_edit_distance_optimality(self):
        for _ in trange(200):
            str_len = random.randint(10, 30)
            str_from = ''.join(random.choices(LETTERS, k=str_len))
            num_ground_truth_edits: int = 0
            str_to = ''
            for i in range(str_len):
                while random.random() < 0.2:
                    # 20% chance of inserting a new character
                    str_to += random.choice(LETTERS)
                    num_ground_truth_edits += 1
                num_ground_truth_edits += 1
                if random.random() < 0.2:
                    # 20% chance of removing the original character
                    pass
                else:
                    str_to += str_from[i]
            distance: EditDistance = string_edit_distance(str_from, str_to)
            edits: List[Edit] = list(distance.edits())
            num_edits = len(edits)
            if num_ground_truth_edits < num_edits:
                print()
                print('\n'.join([e.__class__.__name__ for e in edits]))
                print(str_from, str_to)
            self.assertGreaterEqual(num_ground_truth_edits, num_edits)

    def test_empty_string_edit_distance(self):
        with self.assertRaises(StopIteration):
            next(string_edit_distance('', '').edits())
        self.assertEqual(
            3,
            sum(1 for _ in string_edit_distance('foo', '').edits())
        )
        self.assertEqual(
            3,
            sum(1 for _ in string_edit_distance('', 'foo').edits())
        )
