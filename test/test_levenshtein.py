import random
from unittest import TestCase

from tqdm import trange

from graphtage import EditDistance, string_edit_distance
from graphtage.edits import Edit, Insert, Match, Remove
from graphtage.levenshtein import levenshtein_distance

LETTERS: str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
SMALL_ALPHABET: str = 'abcd'


def render_script(distance: EditDistance) -> list[str]:
    """Renders the edit script that a string :class:`EditDistance` reconstructs.

    Each edit becomes one token: ``x>y`` for a match of ``x`` against ``y`` (a substitution when the two
    differ), ``+y`` for an insertion, and ``-x`` for a removal.

    Args:
        distance: the edit whose script to render.

    Returns:
        list[str]: One token per edit, in the order the edits are emitted.

    """
    script: list[str] = []
    for edit in distance.edits():
        if isinstance(edit, Match):
            script.append(f"{edit.from_node.object}>{edit.to_node.object}")
        elif isinstance(edit, Remove):
            script.append(f"-{edit.from_node.object}")
        elif isinstance(edit, Insert):
            script.append(f"+{edit.from_node.object}")
        else:
            raise AssertionError(f"Unexpected edit type {edit.__class__.__name__} in a string edit script")
    return script


def edit_script(from_str: str, to_str: str) -> list[str]:
    """Renders the edit script that :func:`graphtage.string_edit_distance` reconstructs for two strings.

    Args:
        from_str: the string to match from.
        to_str: the string to match to.

    Returns:
        list[str]: The tokens described by :func:`render_script`.

    """
    return render_script(string_edit_distance(from_str, to_str))


def replay_script(script: list[str]) -> tuple[str, str, int]:
    """Replays a script rendered by :func:`edit_script`.

    Args:
        script: the tokens to replay.

    Returns:
        tuple[str, str, int]: The string the script matches from, the string it matches to, and its total cost.

    """
    from_str, to_str, cost = '', '', 0
    for token in script:
        if token[0] == '+':
            to_str += token[1]
            cost += 1
        elif token[0] == '-':
            from_str += token[1]
            cost += 1
        else:
            from_str += token[0]
            to_str += token[2]
            cost += int(token[0] != token[2])
    return from_str, to_str, cost


def optimal_alignment(from_str: str, to_str: str) -> tuple[int, int]:
    """Scores the best alignment of two strings, independently of :class:`graphtage.EditDistance`.

    Alignments are ordered the way :meth:`graphtage.levenshtein.EditDistance._best_match` orders them: by total
    edit cost first, then by the number of edits. A substitution, an insertion, and a removal each cost one.

    Args:
        from_str: the string to align from.
        to_str: the string to align to.

    Returns:
        tuple[int, int]: The cost of the best alignment and the number of edits in it.

    """
    rows, cols = len(from_str) + 1, len(to_str) + 1
    best: list[list[tuple[int, int]]] = [[(0, 0)] * cols for _ in range(rows)]
    for row in range(1, rows):
        best[row][0] = (row, row)
    for col in range(1, cols):
        best[0][col] = (col, col)
    for row in range(1, rows):
        for col in range(1, cols):
            diagonal, up, left = best[row - 1][col - 1], best[row - 1][col], best[row][col - 1]
            best[row][col] = min(
                (diagonal[0] + int(from_str[row - 1] != to_str[col - 1]), diagonal[1] + 1),
                (up[0] + 1, up[1] + 1),
                (left[0] + 1, left[1] + 1),
            )
    return best[-1][-1]


class TestEditDistance(TestCase):
    def test_string_edit_distance_reconstruction(self):
        for _ in trange(200):
            str1_len = random.randint(10, 30)
            str2_len = random.randint(10, 30)
            str_from = ''.join(random.choices(LETTERS, k=str1_len))
            str_to = ''.join(random.choices(LETTERS, k=str2_len))
            distance: EditDistance = string_edit_distance(str_from, str_to)
            edits: list[Edit] = list(distance.edits())
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
            edits: list[Edit] = list(distance.edits())
            num_edits = len(edits)
            if num_ground_truth_edits < num_edits:
                print()
                print('\n'.join([e.__class__.__name__ for e in edits]))
                print(str_from, str_to)
            self.assertGreaterEqual(num_ground_truth_edits, num_edits)

    def test_string_edit_distance_is_levenshtein(self):
        """Cross-checks the edit matrix against the canonical Levenshtein implementation.

        A small alphabet and short strings are used deliberately: they maximize the number of cells in
        which a substitution ties with an insertion paired with a removal, which is the case that
        https://github.com/trailofbits/graphtage/issues/89 got wrong.

        """
        for _ in trange(200):
            str_from = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 10)))
            str_to = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 10)))
            distance: EditDistance = string_edit_distance(str_from, str_to)
            while distance.tighten_bounds():
                pass
            bounds = distance.bounds()
            self.assertTrue(bounds.definitive(), f"{str_from!r} -> {str_to!r} has bounds {bounds!s}")
            self.assertEqual(
                levenshtein_distance(str_from, str_to),
                bounds.upper_bound,
                f"{str_from!r} -> {str_to!r}"
            )

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

    def assert_edit_script(self, from_str: str, to_str: str, expected: list[str]):
        """Asserts that the edit script for a pair of strings is exactly ``expected``.

        The expectation is first checked against :func:`optimal_alignment`, so a script that is merely what the
        current code emits cannot be pinned as a contract.

        Args:
            from_str: the string to match from.
            to_str: the string to match to.
            expected: the tokens that :func:`edit_script` is expected to produce.

        """
        replayed_from, replayed_to, cost = replay_script(expected)
        pair = f"{from_str!r} -> {to_str!r}"
        self.assertEqual(from_str, replayed_from, f"{pair}: the expected script does not match from {from_str!r}")
        self.assertEqual(to_str, replayed_to, f"{pair}: the expected script does not match to {to_str!r}")
        self.assertEqual(
            optimal_alignment(from_str, to_str),
            (cost, len(expected)),
            f"{pair}: the expected script {expected!r} is not an optimal alignment"
        )
        self.assertEqual(expected, edit_script(from_str, to_str), pair)

    def test_edit_script_tie_break_is_stable(self):
        """Pins the edit script for pairs of strings that have more than one optimal alignment.

        :meth:`graphtage.levenshtein.EditDistance._best_match` orders candidate predecessors by accumulated cost,
        then by the number of edits on the path, then by a fixed direction order: the diagonal, then the border
        insertion, then the border removal. The second key is what makes one substitution beat an insertion paired
        with a removal of the same cost. The direction order is what makes removals come out before insertions,
        because reconstruction walks the matrix backwards.

        Nothing else in the suite checks *which* optimal alignment is chosen, only that some optimal alignment is.
        These expectations fail if the candidate order in ``_best_match`` is permuted or if the path-length key is
        dropped, both of which leave the total cost optimal and the rendered diff different.

        """
        self.assert_edit_script('', 'a', ['+a'])
        self.assert_edit_script('foo', '', ['-f', '-o', '-o'])
        self.assert_edit_script('ab', 'ba', ['a>b', 'b>a'])
        self.assert_edit_script('abc', 'acb', ['a>a', 'b>c', 'c>b'])
        self.assert_edit_script('aa', 'aba', ['a>a', '+b', 'a>a'])
        # Two substitutions and a removal tie on cost with a removal, two matches and an insertion; the
        # path-length key picks the shorter script.
        self.assert_edit_script('aabc', 'bcb', ['a>b', 'a>c', 'b>b', '-c'])
        # The two border directions tie on both keys; the removal has to come out before the insertion.
        self.assert_edit_script('aba', 'bab', ['-a', 'b>b', 'a>a', '+b'])
        # Six alignments tie on both keys, so every one of the three directions is load-bearing here.
        self.assert_edit_script('caccda', 'bcddcb', ['-c', 'a>b', 'c>c', 'c>d', 'd>d', '+c', 'a>b'])
