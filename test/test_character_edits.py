"""Character edit scripts computed directly must match the ones built by tightening the lattice."""
import random
from io import StringIO
from unittest import TestCase

from graphtage import string_edit_distance
from graphtage.printer import NULL_PRINTER, Printer, get_default_printer, set_default_printer


def script(distance):
    return [(type(e).__name__, repr(e.from_node), repr(e.to_node), repr(e.bounds())) for e in distance.edits()]


class TestCharacterEdits(TestCase):
    def setUp(self):
        self.printer = get_default_printer()

    def tearDown(self):
        set_default_printer(self.printer)

    def assert_matches_incremental_refinement(self, before: str, after: str):
        direct = string_edit_distance(before, after)
        incremental = string_edit_distance(before, after)
        # Tightening first starts the fringe, so edits() takes the general path for this one.
        while incremental.tighten_bounds():
            pass
        self.assertEqual(script(incremental), script(direct))
        self.assertEqual(incremental.bounds(), direct.bounds())
        self.assertEqual(incremental._last_fringe, direct._last_fringe)
        self.assertEqual(incremental._fringe_row, direct._fringe_row)
        self.assertEqual(incremental._fringe_col, direct._fringe_col)
        self.assertIsNone(direct.edit_matrix)
        self.assertIsNone(direct.path_costs)

    def test_completion_matches_incremental_refinement(self):
        rng = random.Random(441)
        for printer in (NULL_PRINTER, Printer(out_stream=StringIO())):
            set_default_printer(printer)
            for _ in range(100):
                before, after = (''.join(rng.choices('abé\n', k=rng.randrange(20))) for _ in range(2))
                with self.subTest(quiet=printer.quiet, before=before, after=after):
                    self.assert_matches_incremental_refinement(before, after)

    def test_ties_match_incremental_refinement(self):
        # A two-letter alphabet produces many equally cheap alignments, which exercises the tie order.
        set_default_printer(NULL_PRINTER)
        rng = random.Random(7)
        for _ in range(200):
            before, after = (''.join(rng.choices('ab', k=rng.randrange(1, 12))) for _ in range(2))
            with self.subTest(before=before, after=after):
                self.assert_matches_incremental_refinement(before, after)
