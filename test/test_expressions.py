from unittest import TestCase

from graphtage.expressions import parse, ParseError, StringToken


class TestExpressions(TestCase):
    def test_string_parsing(self):
        input_str = 'This is a test'
        ret = parse(f'"{input_str}"').eval()
        self.assertIsInstance(ret, StringToken)
        self.assertEqual(input_str, str(ret))

    def test_string_escaping(self):
        input_str = 'foo " bar'
        escaped_input = input_str.replace('"', '\\"')
        ret = parse(f'"{escaped_input}"').eval()
        self.assertIsInstance(ret, StringToken)
        self.assertEqual(input_str, str(ret))
        with self.assertRaises(ParseError):
            parse(f'{input_str}')

    def test_getitem(self):
        self.assertEqual(1234, parse('foo[(bar + 10) * 2]').eval({
            'foo': {
                40: 1234
            },
            'bar': 10
        }))

    def test_bracket_parsing(self):
        with self.assertRaises(ParseError):
            parse('foo[bar(])')
        with self.assertRaises(ParseError):
            parse('(bar[)]')

    def test_evaluation(self):
        assignments = {
            'sampling_factors': 1234,
            'thumbnail_x': 5,
            'thumbnail_y': 7
        }
        self.assertEqual(65, parse('(sampling_factors & -0xf0) >> 4').eval(assignments))
        self.assertEqual(105, parse('thumbnail_x * thumbnail_y * 3').eval(assignments))

    def test_functions(self):
        self.assertEqual(sum([1, 2, 3, 4]), parse('sum([1, 2, 3, 4])').eval())
        self.assertEqual('a, b, c, d', parse('", ".join(["a", "b", "c", "d"])').eval())

    def test_member_access(self):
        class Foo:
            def __init__(self, bar):
                self.bar = bar

        assignments = {
            'foo': Foo(1234)
        }

        self.assertEqual(1234, parse('foo.bar').eval(assignments))
        with self.assertRaises(ParseError):
            parse('foo.__dict__').eval(assignments)

    def test_containers(self):
        self.assertEqual([[1, (3,)]], parse('[[1, (3,)]]').eval())
        self.assertEqual([1, 2, 3, 4], parse('[1, 2, 3, 4]').eval())
        self.assertEqual((1, 2, 3, 4), parse('(1, 2, 3, 4)').eval())
        self.assertEqual([[1, 2, [3], 4]], parse('[[1, 2, [3], 4]]').eval())
        self.assertEqual((1,), parse('(1,)').eval())
        self.assertEqual([1], parse('[1]').eval())
        with self.assertRaises(ParseError):
            self.assertEqual([1], parse('[1,]').eval())
