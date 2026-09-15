import os
import random
import time
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from graphtage.batch_distance import (
    BACKEND_ENV_VAR,
    NUMPY_MIN_PAIRS,
    NumpyBackend,
    all_flat,
    all_pairs,
    available_backends,
    register_backend,
)
from graphtage.levenshtein import levenshtein_distance

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
UNICODE_ALPHABET = 'a\u00e9\u6f22\u00df\u0301\U0001F600\U00010330'


def oracle_pairs(from_strings, to_strings) -> np.ndarray:
    """Computes the same grid as :func:`graphtage.batch_distance.all_pairs`, one pair at a time."""
    return np.array(
        [[levenshtein_distance(left, right) for right in to_strings] for left in from_strings],
        dtype=np.int64,
    ).reshape(len(from_strings), len(to_strings))


def random_strings(rng: random.Random, count: int, alphabet: str, shortest: int, longest: int) -> list[str]:
    """Draws ``count`` strings of between ``shortest`` and ``longest`` symbols from ``alphabet``."""
    return [''.join(rng.choices(alphabet, k=rng.randint(shortest, longest))) for _ in range(count)]


def mixed_corpus() -> list[str]:
    """Builds the corpus that the backend agreement tests run over.

    It covers every case that has its own code path: the empty string, single characters, strings that repeat, a
    long string and a near copy of it, non-ASCII text including a combining mark, and characters outside the Basic
    Multilingual Plane.

    """
    rng = random.Random(0xC0FFEE)
    long_string = ''.join(rng.choices(ALPHABET, k=200))
    return [
        '',
        'a',
        'b',
        'a',
        'ab',
        long_string,
        long_string[:100] + 'Z' + long_string[101:],
        'h\u00e9llo w\u00f6rld',
        'h\u00e9llo world',
        'e\u0301clair',
        '\u00e9clair',
        '\U0001F600\U0001F601\U0001F602',
        '\U0001F600x\U0001F602',
        '\U00010330\U00010331',
        *random_strings(rng, 8, ALPHABET, 0, 14),
        *random_strings(rng, 6, UNICODE_ALPHABET, 1, 9),
    ]


class TestBackendAgreement(TestCase):
    def test_backends_agree_with_levenshtein_distance(self):
        """Every backend returns exactly what ``levenshtein_distance`` returns, over a varied corpus.

        Prevents any backend from computing a different metric than the one Graphtage already uses: a wrong
        substitution cost, a wrong insertion or deletion cost, or a Unicode encoding that splits or merges
        characters. It also covers the trivial answers that never reach a backend, because the corpus contains
        empty strings and repeated strings.

        """
        corpus = mixed_corpus()
        expected = oracle_pairs(corpus, corpus)
        for backend in available_backends():
            with self.subTest(backend=backend):
                self.assertTrue(np.array_equal(all_pairs(corpus, corpus, backend=backend), expected))

    def test_backends_agree_pairwise(self):
        """``all_flat`` answers each position with the distance of the pair at that position.

        Prevents the pairwise entry point from transposing its arguments or from losing the mapping back from the
        deduplicated pair list to the input positions. Pairs deliberately repeat, in a shuffled order, so that the
        deduplicated list is both shorter than the input and in a different order.

        """
        rng = random.Random(4242)
        distinct = list(
            zip(random_strings(rng, 24, ALPHABET, 0, 18), random_strings(rng, 24, ALPHABET, 0, 18), strict=True)
        )
        pairs = [rng.choice(distinct) for _ in range(72)]
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        expected = np.array([levenshtein_distance(a, b) for a, b in zip(left, right, strict=True)], dtype=np.int64)
        for backend in available_backends():
            with self.subTest(backend=backend):
                self.assertTrue(np.array_equal(all_flat(left, right, backend=backend), expected))

    def test_astral_plane_characters_are_single_symbols(self):
        """A character outside the Basic Multilingual Plane counts as one edit, not two.

        Prevents an encoding that measures UTF-16 code units or UTF-8 bytes instead of code points. Python iterates
        a ``str`` by code point, so ``'\U0001F600'`` has to cost the same single substitution that ``'a'`` does.

        """
        self.assertEqual(all_pairs(['\U0001F600'], ['a'], backend='numpy')[0, 0], 1)
        self.assertEqual(all_pairs(['\U0001F600'], [''], backend='numpy')[0, 0], 1)
        self.assertEqual(all_pairs(['x\U0001F600y'], ['xy'], backend='numpy')[0, 0], 1)
        astral = ['\U0001F600\U0001F601', '\U0001F600a', 'ab', '\U00010330b', 'b\U0001F601']
        self.assertTrue(np.array_equal(all_pairs(astral, astral, backend='numpy'), oracle_pairs(astral, astral)))

    def test_readout_uses_each_pairs_own_length(self):
        """Each pair is read out at its own string lengths, not at the padded width of its batch.

        Prevents a silent off-by-one in the padding mask. Pairs of different lengths share one batch, so reading a
        pair's answer one column early, or at the width the batch was padded to, returns a cell that describes a
        padded string rather than the real one.

        """
        rng = random.Random(90210)
        from_strings = random_strings(rng, 9, ALPHABET, 6, 6)
        to_strings = [''.join(rng.choices(ALPHABET, k=length)) for length in range(1, 12)]
        self.assertGreater(len(from_strings) * len(to_strings), NUMPY_MIN_PAIRS)
        self.assertTrue(
            np.array_equal(
                all_pairs(from_strings, to_strings, backend='numpy'),
                oracle_pairs(from_strings, to_strings),
            )
        )

    def test_repeated_strings_map_back_to_every_position(self):
        """A string that appears more than once gets its answer at each of its positions.

        Prevents the deduplication from scattering the distance grid back through the wrong indices, which an
        unbalanced grid of repeated values exposes and a square one of distinct values does not.

        """
        from_strings = ['aa', 'b', 'aa', 'cccc', 'b', 'b']
        to_strings = ['b', 'aa', 'b', 'dd']
        self.assertTrue(
            np.array_equal(
                all_pairs(from_strings, to_strings, backend='numpy'),
                oracle_pairs(from_strings, to_strings),
            )
        )


class TestLengthSkew(TestCase):
    def test_one_long_string_does_not_slow_down_the_batch(self):
        """A single long string among short ones is bucketed away from them.

        Prevents the batch from being padded to its longest member. Without bucketing, every short pair would be
        computed over a 4 KB wide row, which measured about 30 times slower than the bucketed run and, more
        tellingly, slower than the pure Python backend it is supposed to replace. The bound is relative to the
        ``python`` backend measured on the same machine in the same run, so it does not depend on how fast the
        machine is.

        """
        rng = random.Random(1861)
        from_strings = [''.join(rng.choices(ALPHABET, k=4096)), *random_strings(rng, 60, ALPHABET, 5, 5)]
        to_strings = random_strings(rng, 60, ALPHABET, 5, 5)

        start = time.perf_counter()
        batched = all_pairs(from_strings, to_strings, backend='numpy')
        batched_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scalar = all_pairs(from_strings, to_strings, backend='python')
        scalar_seconds = time.perf_counter() - start

        self.assertTrue(np.array_equal(batched, scalar))
        self.assertTrue(np.array_equal(batched, oracle_pairs(from_strings, to_strings)))
        self.assertLess(
            batched_seconds * 3,
            scalar_seconds,
            f"the numpy backend took {batched_seconds:.4f}s against the python backend's {scalar_seconds:.4f}s, "
            f"which suggests the batch was padded to its longest string",
        )


class TestEmptyInputs(TestCase):
    def test_empty_inputs_have_the_right_shape(self):
        """An empty collection yields an empty axis rather than an error or a squeezed result.

        Prevents the reshape and the index-mapping from collapsing a zero-length axis, which would make a caller
        that indexes the result by position fail somewhere far from the cause.

        """
        self.assertEqual(all_pairs([], []).shape, (0, 0))
        self.assertEqual(all_pairs(['a'], []).shape, (1, 0))
        self.assertEqual(all_pairs([], ['a']).shape, (0, 1))
        self.assertEqual(all_flat([], []).shape, (0,))

    def test_empty_strings_cost_the_other_side(self):
        """The distance to or from the empty string is the length of the other side.

        Prevents the short circuit that keeps empty strings out of the batch from returning zero, and prevents an
        empty string from reaching the scan, whose row loop assumes at least one symbol.

        """
        self.assertTrue(np.array_equal(all_pairs([''], ['', 'a', 'abc'])[0], np.array([0, 1, 3])))
        self.assertTrue(np.array_equal(all_pairs(['', 'a', 'abc'], [''])[:, 0], np.array([0, 1, 3])))

    def test_all_flat_rejects_collections_of_different_lengths(self):
        """Pairing up two collections of different lengths is an error rather than a truncated answer."""
        with self.assertRaises(ValueError):
            all_flat(['a', 'b'], ['a'])


class TestBackendSelection(TestCase):
    def test_environment_variable_pins_the_backend(self):
        """``GRAPHTAGE_BATCH_BACKEND`` selects a backend, and every choice gives the same answer.

        Prevents the environment override from changing the result rather than only the route to it, which is what
        makes it safe to benchmark one backend against another.

        """
        rng = random.Random(2718)
        corpus = random_strings(rng, 8, ALPHABET, 1, 12)
        expected = oracle_pairs(corpus, corpus)
        for backend in available_backends():
            with self.subTest(backend=backend), patch.dict(os.environ, {BACKEND_ENV_VAR: backend}):
                self.assertTrue(np.array_equal(all_pairs(corpus, corpus), expected))

    def test_environment_variable_overrides_the_automatic_choice(self):
        """A batch that would go to ``numpy`` goes to ``python`` when the environment says so.

        Prevents the environment variable from being read but ignored. Both backends return the same numbers, so a
        result comparison cannot tell them apart; this watches which one is asked.

        """
        rng = random.Random(1414)
        corpus = random_strings(rng, 12, ALPHABET, 4, 12)
        self.assertGreater(len(corpus) ** 2, NUMPY_MIN_PAIRS)
        with patch.object(NumpyBackend, 'distances', autospec=True, side_effect=NumpyBackend.distances) as batched:
            with patch.dict(os.environ, {BACKEND_ENV_VAR: 'python'}):
                all_pairs(corpus, corpus)
            batched.assert_not_called()
            all_pairs(corpus, corpus)
            self.assertTrue(batched.called, 'this batch should have gone to the numpy backend on its own')

    def test_environment_variable_names_are_validated(self):
        """A misspelled backend name in the environment fails loudly instead of being ignored."""
        with patch.dict(os.environ, {BACKEND_ENV_VAR: 'does-not-exist'}), self.assertRaises(ValueError):
            all_pairs(['a'], ['b'])

    def test_small_batches_fall_back(self):
        """A batch below every threshold still gets an answer, from the fallback backend."""
        self.assertLess(1, NUMPY_MIN_PAIRS)
        self.assertEqual(all_pairs(['kitten'], ['sitting'])[0, 0], 3)

    def test_unknown_backend_is_rejected(self):
        """Naming a backend that is not registered fails immediately and says what is available."""
        with self.assertRaises(ValueError) as context:
            all_pairs(['a'], ['b'], backend='does-not-exist')
        self.assertIn('does-not-exist', str(context.exception))

    def test_a_name_cannot_be_registered_twice(self):
        """Registering over an existing backend fails rather than silently replacing it."""
        with self.assertRaises(ValueError):
            register_backend(NumpyBackend())

    def test_available_backends_are_ordered_fastest_first(self):
        """The registry reports the backends in the order the dispatcher would prefer them."""
        backends = available_backends()
        self.assertIn('python', backends)
        self.assertIn('numpy', backends)
        self.assertLess(backends.index('numpy'), backends.index('python'))


class TestBytes(TestCase):
    def test_bytes_agree_with_levenshtein_distance(self):
        """Byte strings are measured by byte, the same way ``levenshtein_distance`` measures them."""
        rng = random.Random(31337)
        corpus = [value.encode('utf-8') for value in random_strings(rng, 12, UNICODE_ALPHABET, 0, 10)]
        expected = oracle_pairs(corpus, corpus)
        for backend in available_backends():
            with self.subTest(backend=backend):
                self.assertTrue(np.array_equal(all_pairs(corpus, corpus, backend=backend), expected))

    def test_a_pair_cannot_mix_str_and_bytes(self):
        """Comparing a ``str`` to a ``bytes`` is rejected instead of answered.

        Prevents the encoding collision the two share: ``ord('a')`` and ``b'a'[0]`` are both 97, so a batch that
        held both would report ``'a'`` and ``b'a'`` as equal, while Python reports them as different. There is no
        answer that satisfies both readings, so neither is offered.

        """
        for left, right in ((['a'], [b'a']), ([b'a'], ['a']), (['abc'], [b'abd'])):
            with self.subTest(left=left, right=right), self.assertRaises(TypeError):
                all_pairs(left, right)

    def test_str_and_bytes_batches_stay_separate(self):
        """A call holding both kinds gives each kind the answer it would get on its own.

        Prevents the split into a ``str`` sub-batch and a ``bytes`` sub-batch from scattering its results back to
        the wrong input positions, which is the bookkeeping that keeps the two encodings apart. The two halves are
        drawn from the same lengths, so neither is a prefix of the sorted batch and a misplaced scatter shows up.

        """
        rng = random.Random(1234321)
        text = random_strings(rng, 20, ALPHABET, 1, 6)
        blobs = [value.encode('ascii') for value in random_strings(rng, 20, ALPHABET, 1, 6)]
        left = [*text, *blobs]
        right = [*random_strings(rng, 20, ALPHABET, 1, 6),
                 *[value.encode('ascii') for value in random_strings(rng, 20, ALPHABET, 1, 6)]]
        self.assertTrue(
            np.array_equal(
                all_flat(left, right, backend='numpy'),
                np.array([levenshtein_distance(a, b) for a, b in zip(left, right, strict=True)], dtype=np.int64),
            )
        )

    def test_other_types_are_rejected(self):
        """Anything that is neither a ``str`` nor a ``bytes`` fails with a message naming its type."""
        with self.assertRaises(TypeError) as context:
            all_pairs([1], [2])
        self.assertIn('int', str(context.exception))
