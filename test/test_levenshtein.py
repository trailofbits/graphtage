import random
from unittest import TestCase

from tqdm import trange

import graphtage
from graphtage import EditDistance, batch_distance, string_edit_distance
from graphtage.edits import Edit, Insert, Match, Remove
from graphtage.levenshtein import exact_string_distance, levenshtein_distance

LETTERS: str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
SMALL_ALPHABET: str = 'abcd'
NON_ASCII: str = 'αβγδεζηθ🙂é́'


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

    def test_shared_prefix_biases_the_alignment(self):
        """Pins the edit scripts that the shared-prefix strip in :meth:`EditDistance.__init__` decides.

        Stripping a common prefix reads as a pure optimization, but it is output-visible: it forces the leading
        characters to be matched diagonally, whereas backward reconstruction otherwise reaches the origin by a
        border move. Roughly 10% of small-alphabet pairs produce a different — equally optimal — script when the
        strip is removed, which is why these expectations exist.

        The shared-suffix strip has no such effect, because it agrees with the diagonal-first tie-break that
        reconstruction already applies at the end of the matrix.

        """
        self.assert_edit_script('a', 'aa', ['a>a', '+a'])
        self.assert_edit_script('cc', 'c', ['c>c', '-c'])
        self.assert_edit_script('aac', 'ab', ['a>a', '-a', 'c>b'])
        self.assert_edit_script('cbcbbb', 'caa', ['c>c', '-b', '-c', '-b', 'b>a', 'b>a'])

    def test_edit_script_realizes_the_reported_cost(self):
        """Checks that the reconstructed script costs what the edit reports as its distance.

        :meth:`TestEditDistance.test_string_edit_distance_is_levenshtein` checks the reported distance and
        :meth:`TestEditDistance.test_string_edit_distance_reconstruction` checks that the script rebuilds both
        strings, but nothing checks that the script a user sees adds up to the cost the edit reports. A cost
        computed anywhere other than from the script itself passes both of the older tests.

        """
        for _ in trange(200):
            str_from = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 10)))
            str_to = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 10)))
            distance: EditDistance = string_edit_distance(str_from, str_to)
            script = render_script(distance)
            replayed_from, replayed_to, cost = replay_script(script)
            pair = f"{str_from!r} -> {str_to!r}"
            self.assertEqual(str_from, replayed_from, pair)
            self.assertEqual(str_to, replayed_to, pair)
            self.assertEqual(levenshtein_distance(str_from, str_to), cost, pair)
            self.assertEqual(optimal_alignment(str_from, str_to), (cost, len(script)), pair)
            bounds = distance.bounds()
            self.assertTrue(bounds.definitive(), f"{pair} has bounds {bounds!s}")
            self.assertEqual(cost, bounds.upper_bound, pair)


class TestExactStringDistance(TestCase):
    """Covers :func:`graphtage.levenshtein.exact_string_distance`.

    The function short-circuits equal and empty operands and strips a shared prefix and suffix before calling
    :func:`graphtage.levenshtein.levenshtein_distance`. Every one of those steps is an opportunity to return a
    number that is not the Levenshtein distance, and the number it returns is the cost of a
    :class:`graphtage.StringEdit`, so a wrong answer silently changes which nodes a diff matches.

    """

    def assert_agrees(self, s, t):
        expected = levenshtein_distance(s, t)
        pair = f"{s!r} -> {t!r}"
        self.assertEqual(expected, exact_string_distance(s, t), pair)
        self.assertEqual(expected, exact_string_distance(t, s), f"{t!r} -> {s!r}")

    def test_agrees_on_random_strings(self):
        """A small alphabet maximizes the number of pairs with a shared prefix or suffix to strip."""
        for _ in trange(500):
            s = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 12)))
            t = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 12)))
            self.assert_agrees(s, t)

    def test_agrees_on_random_bytes(self):
        """:class:`StringNode` wraps ``bytes`` as well as ``str``, and indexing ``bytes`` yields ``int``."""
        for _ in trange(500):
            s = bytes(random.choices(range(0, 8), k=random.randint(0, 12)))
            t = bytes(random.choices(range(0, 8), k=random.randint(0, 12)))
            self.assert_agrees(s, t)

    def test_agrees_on_random_non_ascii(self):
        """Astral characters and combining marks are single positions to both implementations."""
        for _ in trange(200):
            s = ''.join(random.choices(NON_ASCII, k=random.randint(0, 10)))
            t = ''.join(random.choices(NON_ASCII, k=random.randint(0, 10)))
            self.assert_agrees(s, t)

    def test_empty_operands(self):
        self.assertEqual(0, exact_string_distance('', ''))
        self.assertEqual(0, exact_string_distance(b'', b''))
        self.assertEqual(3, exact_string_distance('', 'abc'))
        self.assertEqual(3, exact_string_distance('abc', ''))
        self.assertEqual(3, exact_string_distance(b'', b'abc'))
        self.assertEqual(3, exact_string_distance(b'abc', b''))

    def test_shared_prefix_and_suffix_do_not_overlap(self):
        """An operand that is wholly a prefix of the other must not have its characters counted twice."""
        self.assertEqual(1, exact_string_distance('aa', 'aaa'))
        self.assertEqual(2, exact_string_distance('aaa', 'aaaaa'))
        self.assertEqual(4, exact_string_distance('a', 'aaaaa'))
        self.assertEqual(1, exact_string_distance('ab', 'aab'))
        self.assertEqual(0, exact_string_distance('aaaa', 'aaaa'))

    def test_str_never_equals_bytes(self):
        """A mixed pair costs one per aligned position, which is what the character lattice charges."""
        self.assertEqual(5, exact_string_distance('hello', b'hello'))
        self.assertEqual(5, exact_string_distance(b'hello', 'hello'))
        self.assertEqual(1, exact_string_distance('a', b'a'))

    def test_agrees_with_the_lattice(self):
        """The lattice is what renders the edit, so the reported cost has to be the cost of its script."""
        for _ in trange(100):
            s = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 10)))
            t = ''.join(random.choices(SMALL_ALPHABET, k=random.randint(0, 10)))
            lattice = string_edit_distance(s, t)
            while lattice.tighten_bounds():
                pass
            self.assertEqual(lattice.bounds().upper_bound, exact_string_distance(s, t), f"{s!r} -> {t!r}")


class TestPrePricing(TestCase):
    """Covers the ``preprice`` argument of :class:`graphtage.levenshtein.EditDistance`."""

    def setUp(self):
        batch_distance.clear()

    def tearDown(self):
        batch_distance.clear()

    def test_a_sequence_of_leaves_is_pre_priced(self):
        """The cross product of two sequences of leaves is priced in one batch before any cell is built.

        Prevents the pre-pricing call from being dropped from the constructor, which no comparison of results can
        detect because both routes return the same distances.

        """
        from_node = graphtage.json.build_tree([f"item number {index}" for index in range(12)])
        to_node = graphtage.json.build_tree([f"item no. {index}" for index in range(12)])
        self.assertIsInstance(from_node.edits(to_node), EditDistance)
        self.assertTrue(batch_distance._blocks)

    def test_a_character_lattice_is_not_pre_priced(self):
        """The lattice over the characters of two strings prices nothing in advance.

        Prevents the character level :class:`EditDistance` from paying for a collection pass over every cell of
        every string a diff renders, in exchange for a batch of single-character pairs whose distances are
        settled by the equality short circuit before any dynamic program runs.

        """
        lattice = string_edit_distance('kittens are nice', 'sitting is nicer')
        while lattice.tighten_bounds():
            pass
        self.assertEqual((), batch_distance._blocks)
