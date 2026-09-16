import os
import random
import time
from io import StringIO
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import graphtage
from graphtage import batch_distance
from graphtage.batch_distance import (
    BACKEND_ENV_VAR,
    NUMPY_MIN_PAIRS,
    NumpyBackend,
    all_flat,
    all_pairs,
    available_backends,
    cost,
    register_backend,
)
from graphtage.levenshtein import levenshtein_distance
from graphtage.printer import Printer

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


class TestCostOracle(TestCase):
    """Tests for :func:`graphtage.batch_distance.cost` and the blocks it reads."""

    def setUp(self):
        batch_distance.clear()

    def tearDown(self):
        batch_distance.clear()

    def assert_cost_is_exact(self, from_strings, to_strings):
        """Asserts that :func:`graphtage.batch_distance.cost` answers every pair exactly.

        Args:
            from_strings: The strings to measure from.
            to_strings: The strings to measure to.

        """
        for left in from_strings:
            for right in to_strings:
                with self.subTest(left=left, right=right):
                    self.assertEqual(levenshtein_distance(left, right), cost(left, right))

    def test_cost_matches_levenshtein_distance_without_a_block(self):
        """Every pair gets the same answer the scalar dynamic program gives, with nothing pre-priced.

        Prevents the short circuits and the shared affix stripping in the miss path from changing the metric. The
        corpus holds empty strings, repeated strings, strings that share an affix, non-ASCII text, and characters
        outside the Basic Multilingual Plane.

        """
        self.assert_cost_is_exact(mixed_corpus(), mixed_corpus())

    def test_cost_matches_levenshtein_distance_from_a_block(self):
        """A pre-priced block answers with exactly what computing the pair would have answered.

        Prevents a block from being indexed by the wrong axis, which a square grid of a symmetric metric would
        hide, and prevents its :class:`numpy.int32` cells from being read back as anything but the distance.

        """
        rng = random.Random(51966)
        from_strings = random_strings(rng, 14, ALPHABET, 0, 16)
        to_strings = random_strings(rng, 11, ALPHABET, 0, 16)
        batch_distance.preprice_product(from_strings, to_strings)
        self.assert_cost_is_exact(from_strings, to_strings)

    def test_a_block_answers_instead_of_the_scalar_path(self):
        """A pre-priced pair is answered from the block rather than computed again.

        Prevents the block from being installed but never consulted, which no comparison of answers can detect
        because both paths return the same number.

        """
        rng = random.Random(31337)
        from_strings = random_strings(rng, 8, ALPHABET, 4, 16)
        to_strings = random_strings(rng, 8, ALPHABET, 4, 16)
        batch_distance.preprice_product(from_strings, to_strings)
        with patch.object(batch_distance, '_uncached_cost', side_effect=AssertionError('block was not consulted')):
            self.assert_cost_is_exact(from_strings, to_strings)

    def test_eviction_does_not_change_answers(self):
        """Filling the block stack past its limit drops the oldest blocks without corrupting any answer.

        Prevents an eviction policy that leaves a stale block in place or that drops the wrong end of the stack.
        The pairs of the evicted blocks have to be recomputed, and those answers have to agree with the ones that
        are still cached.

        """
        rng = random.Random(0xBADF00D)
        batches = [
            (random_strings(rng, 7, ALPHABET, 3, 12), random_strings(rng, 7, ALPHABET, 3, 12))
            for _ in range(batch_distance.BLOCK_STACK_LIMIT + 4)
        ]
        for from_strings, to_strings in batches:
            batch_distance.preprice_product(from_strings, to_strings)
        self.assertEqual(batch_distance.BLOCK_STACK_LIMIT, len(batch_distance._blocks))
        for from_strings, to_strings in batches:
            self.assert_cost_is_exact(from_strings, to_strings)

    def test_misses_are_not_memoized(self):
        """Pricing a pair that no block holds leaves the cache exactly as it was.

        Prevents the cache from growing with the number of distinct pairs a diff asks about, which is the whole
        cross product, and which nothing would ever evict.

        """
        rng = random.Random(8675309)
        for left, right in zip(
                random_strings(rng, 200, ALPHABET, 1, 12),
                random_strings(rng, 200, ALPHABET, 1, 12),
                strict=True,
        ):
            cost(left, right)
        self.assertEqual((), batch_distance._blocks)

    def test_clear_discards_every_block(self):
        """Clearing the cache leaves the answers alone and the stack empty."""
        rng = random.Random(24601)
        from_strings = random_strings(rng, 8, ALPHABET, 2, 10)
        to_strings = random_strings(rng, 8, ALPHABET, 2, 10)
        batch_distance.preprice_product(from_strings, to_strings)
        self.assertEqual(1, len(batch_distance._blocks))
        batch_distance.clear()
        self.assertEqual((), batch_distance._blocks)
        self.assert_cost_is_exact(from_strings, to_strings)

    def test_pre_pricing_a_flat_list_leaves_unasked_pairs_absent(self):
        """A pair the caller never offered is not read out of the rectangle its sides happen to span.

        Prevents the unfilled cells of a sparse block from being mistaken for distances of zero, which is what a
        zero-filled matrix would report for every pair the caller did not ask about.

        """
        batch_distance.preprice([('kitten', 'sitting'), ('flaw', 'lawn')])
        self.assertEqual(3, cost('kitten', 'sitting'))
        self.assertEqual(2, cost('flaw', 'lawn'))
        self.assertEqual(levenshtein_distance('kitten', 'lawn'), cost('kitten', 'lawn'))
        self.assertEqual(levenshtein_distance('flaw', 'sitting'), cost('flaw', 'sitting'))

    def test_pairs_that_mix_str_and_bytes_are_left_to_the_scalar_path(self):
        """A block never answers a pair whose sides are of different types, and the pair is still priced.

        Prevents a mixed pair from being given one of the two readings of its symbols: ``ord('a')`` and
        ``b'a'[0]`` are both 97 while ``'a' == b'a'`` is :const:`False`, so the batch declines such a pair and the
        scalar path has to pick it up.

        """
        batch_distance.preprice([('abc', b'abc'), ('abc', 'abd')])
        self.assertEqual(1, cost('abc', 'abd'))
        self.assertEqual(3, cost('abc', b'abc'))
        self.assertEqual(levenshtein_distance('abc', b'abc'), cost('abc', b'abc'))

    def test_bytes_and_str_share_a_block_without_colliding(self):
        """A block holding both kinds of string keeps each kind's answers to itself."""
        batch_distance.preprice_product(['abc', b'abc'], ['abd', b'abxd'])
        self.assertEqual(1, cost('abc', 'abd'))
        self.assertEqual(2, cost(b'abc', b'abxd'))
        self.assertEqual(levenshtein_distance('abc', b'abxd'), cost('abc', b'abxd'))

    def test_an_oversized_rectangle_is_not_pre_priced(self):
        """A batch whose distinct sides span too large a rectangle is left to the scalar path.

        Prevents a sparse batch from allocating a matrix far larger than the number of answers it holds.

        """
        rng = random.Random(4096)
        from_strings = random_strings(rng, 6, ALPHABET, 2, 8)
        to_strings = random_strings(rng, 6, ALPHABET, 2, 8)
        with patch.object(batch_distance, 'PREPRICE_MAX_CELLS', 8):
            batch_distance.preprice_product(from_strings, to_strings)
            batch_distance.preprice(list(zip(from_strings, to_strings, strict=True)))
        self.assertEqual((), batch_distance._blocks)
        self.assert_cost_is_exact(from_strings, to_strings)

    def test_one_big_pair_goes_to_a_vectorized_backend(self):
        """A single pair large enough to pay for the array setup is scanned rather than looped over.

        Prevents the single-pair route from being gated on the number of pairs alone, which would send every long
        pair that no block holds back through the pure Python dynamic program.

        """
        rng = random.Random(1024)
        long_left = ''.join(rng.choices(ALPHABET, k=128))
        long_right = ''.join(rng.choices(ALPHABET, k=128))
        short_left, short_right = 'kitten', 'sitting'
        self.assertGreaterEqual(len(long_left) * len(long_right), batch_distance.VECTORIZED_MIN_CELLS)
        self.assertLess(len(short_left) * len(short_right), batch_distance.VECTORIZED_MIN_CELLS)
        with patch.object(NumpyBackend, 'distances', autospec=True, side_effect=NumpyBackend.distances) as batched:
            self.assertEqual(levenshtein_distance(short_left, short_right), cost(short_left, short_right))
            batched.assert_not_called()
            self.assertEqual(levenshtein_distance(long_left, long_right), cost(long_left, long_right))
            self.assertTrue(batched.called, 'a pair this large should have been scanned')

    def test_the_environment_variable_still_pins_a_single_big_pair(self):
        """``GRAPHTAGE_BATCH_BACKEND`` is honored on the single-pair route as well as on a whole batch."""
        rng = random.Random(2048)
        left = ''.join(rng.choices(ALPHABET, k=96))
        right = ''.join(rng.choices(ALPHABET, k=96))
        with patch.object(NumpyBackend, 'distances', autospec=True, side_effect=NumpyBackend.distances) as batched:
            with patch.dict(os.environ, {BACKEND_ENV_VAR: 'python'}):
                self.assertEqual(levenshtein_distance(left, right), cost(left, right))
            batched.assert_not_called()

    def test_every_backend_pre_prices_the_same_answers(self):
        """Pinning each backend in turn fills a block with identical answers.

        Prevents a backend from being exact on its own entry points but wrong through the block, for instance by
        returning a dtype that the block truncates.

        """
        rng = random.Random(0xFEEDFACE)
        from_strings = random_strings(rng, 9, ALPHABET, 0, 14)
        to_strings = random_strings(rng, 9, ALPHABET, 0, 14)
        expected = oracle_pairs(from_strings, to_strings)
        for backend in available_backends():
            with self.subTest(backend=backend), patch.dict(os.environ, {BACKEND_ENV_VAR: backend}):
                batch_distance.clear()
                batch_distance.preprice_product(from_strings, to_strings)
                answered = np.array(
                    [[cost(left, right) for right in to_strings] for left in from_strings], dtype=np.int64
                )
                self.assertTrue(np.array_equal(answered, expected))


def mutated_words(rng: random.Random, count: int, alphabet: str = ALPHABET) -> tuple[list[str], list[str]]:
    """Draws a collection of strings and a copy of it with one substitution in each."""
    words = random_strings(rng, count, alphabet, 6, 18)
    mutated = []
    for word in words:
        position = rng.randrange(len(word))
        mutated.append(word[:position] + rng.choice(alphabet) + word[position + 1:])
    return words, mutated


def diff_shapes() -> list[tuple[str, object, object, bool]]:
    """Builds the documents that the pre-pricing parity tests diff.

    Every shape is large enough to be pre-priced, and together they cover the ordered and unordered paths, keys as
    well as values, and the kinds of string a leaf can wrap.

    Returns:
        List[Tuple[str, object, object, bool]]: A name, the document to diff from, the document to diff to, and
        whether to build lists as unordered collections.

    """
    rng = random.Random(0x5CA1AB1E)
    words, mutated = mutated_words(rng, 24)
    keys, mutated_keys = mutated_words(rng, 40)
    values, mutated_values = mutated_words(rng, 40)
    non_ascii, mutated_non_ascii = mutated_words(rng, 16, UNICODE_ALPHABET)
    multiline = [f"{left}\n{right}\n" for left, right in zip(words, mutated, strict=True)]
    remultiline = [f"{right}\n{left}\n" for left, right in zip(words, mutated, strict=True)]
    return [
        ('ordered', words, mutated, False),
        ('unordered', words, rng.sample(mutated, len(mutated)), True),
        ('ragged', words, mutated[:17], False),
        ('dict', dict(zip(keys, values, strict=True)), dict(zip(mutated_keys, mutated_values, strict=True)), False),
        ('dict-shared-keys', dict(zip(keys, values, strict=True)), dict(zip(keys, mutated_values, strict=True)),
         False),
        ('bytes', [word.encode() for word in words], [word.encode() for word in mutated], False),
        ('non-ascii', non_ascii, mutated_non_ascii, False),
        ('multiline', multiline, remultiline, False),
        ('nested', [{'k': word} for word in words], [{'k': word} for word in mutated], False),
    ]


class TestPrePricedDiffs(TestCase):
    """Checks that pre-pricing changes how a diff is computed and never what it produces."""

    NOT_PRE_PRICED = frozenset({'nested'})
    """Shapes whose string pairs are spread over many small edits, none of them holding enough to batch.

    A list of single-entry dictionaries is one: the outer sequence holds containers, which contribute no pair, and
    each inner dictionary offers one pair of its own.

    """

    def setUp(self):
        batch_distance.clear()

    def tearDown(self):
        batch_distance.clear()

    @staticmethod
    def render(from_obj, to_obj, unordered: bool) -> tuple[str, int]:
        """Diffs two Python objects and returns the rendered diff and the total cost of its edits.

        Args:
            from_obj: The document to diff from.
            to_obj: The document to diff to.
            unordered: Whether to build lists as unordered collections.

        Returns:
            Tuple[str, int]: The rendered diff, and the sum of the costs of every edit in it.

        """
        options = graphtage.BuildOptions(ignore_list_order=unordered)
        stream = StringIO()
        printer = Printer(out_stream=stream, ansi_color=False, quiet=True)
        diff = graphtage.json.build_tree(from_obj, options).diff(graphtage.json.build_tree(to_obj, options))
        total = sum(
            edit.bounds().upper_bound
            for node in diff.dfs() for edit in node.edit_list if edit.has_non_zero_cost()
        )
        graphtage.FILETYPES_BY_TYPENAME['json'].get_default_formatter().print(printer, diff)
        printer.flush(final=True)
        return stream.getvalue(), total

    def test_pre_pricing_does_not_change_a_diff(self):
        """A pre-priced diff renders the same text at the same cost as one priced a pair at a time.

        Prevents pre-pricing from feeding the matcher a different cost matrix than the scalar path would have,
        which would change which nodes a diff matches. It also prevents the collection pass from projecting a node
        onto a string that the edit it stands for never compares, which would leave the block holding an answer
        for the wrong question.

        """
        for name, from_obj, to_obj, unordered in diff_shapes():
            with self.subTest(shape=name):
                priced = self.render(from_obj, to_obj, unordered)
                if name not in self.NOT_PRE_PRICED:
                    self.assertTrue(batch_distance._blocks, f"the {name} shape never got pre-priced")
                batch_distance.clear()
                with patch.object(batch_distance, 'PREPRICE_MIN_PAIRS', 1 << 40):
                    scalar = self.render(from_obj, to_obj, unordered)
                self.assertEqual((), batch_distance._blocks)
                self.assertEqual(scalar, priced)

    def test_a_pre_priced_diff_reads_its_costs_out_of_the_block(self):
        """A diff whose pairs were pre-priced does not go back through the scalar dynamic program for them.

        Prevents :func:`graphtage.levenshtein.exact_string_distance` from being left wired straight to
        :func:`graphtage.levenshtein.levenshtein_distance`, which would install blocks that nothing ever reads and
        show up as no change at all in what the diff produces. This watches the module level name that
        :mod:`graphtage.batch_distance` does not use, so only the unbatched route trips it.

        """
        words, mutated = mutated_words(random.Random(0xB10CC), 24)
        with patch('graphtage.levenshtein.levenshtein_distance', wraps=levenshtein_distance) as scalar:
            self.render(words, mutated, False)
        self.assertEqual(0, scalar.call_count, 'the pre-priced pairs were priced one at a time as well')

    def test_every_backend_produces_the_same_diff(self):
        """Pinning each backend in turn renders the same diff.

        Prevents a backend from being exact in isolation and wrong once a diff depends on it, and makes
        ``GRAPHTAGE_BATCH_BACKEND`` safe to use for benchmarking.

        """
        for name, from_obj, to_obj, unordered in diff_shapes():
            rendered = {}
            for backend in available_backends():
                with self.subTest(shape=name, backend=backend), patch.dict(os.environ, {BACKEND_ENV_VAR: backend}):
                    batch_distance.clear()
                    rendered[backend] = self.render(from_obj, to_obj, unordered)
            with self.subTest(shape=name):
                self.assertEqual(1, len(set(rendered.values())), f"the backends disagree on the {name} shape")
