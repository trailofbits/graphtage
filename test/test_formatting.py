import json
import random
from io import StringIO
from unittest import TestCase

from tqdm import trange

import graphtage
from graphtage import Filetype
from graphtage.formatter import Formatter

STR_BYTES: str = ''.join([
    chr(i) for i in range(32, 127)
] + ['\n', '\t', '\r'])


FILETYPE_TEST_PREFIX = 'test_'
FILETYPE_TEST_SUFFIX = '_formatting'


def filetype_test(test_func):
    def wrapper(self: 'TestFormatting'):
        name = test_func.__name__
        if not name.startswith(FILETYPE_TEST_PREFIX):
            raise ValueError(f'@filetype_test {name} must start with "{FILETYPE_TEST_PREFIX}"')
        elif not name.endswith(FILETYPE_TEST_SUFFIX):
            raise ValueError(f'@filetype_test {name} must end with "{FILETYPE_TEST_SUFFIX}"')
        filetype_name = name[len(FILETYPE_TEST_PREFIX):-len(FILETYPE_TEST_SUFFIX)]
        if filetype_name not in graphtage.FILETYPES_BY_TYPENAME:
            raise ValueError(f'Filetype "{filetype_name}" for @filetype_test {name} not found in graphtage.FILETYPES_BY_TYPENAME')
        filetype = graphtage.FILETYPES_BY_TYPENAME[filetype_name]
        formatter = filetype.get_default_formatter()

        for _ in trange(1000):
            test_iter = test_func(self)
            orig_obj, str_representation = next(test_iter)
            with graphtage.utils.Tempfile(str_representation) as t:
                tree = filetype.build_tree(t)
                stream = StringIO()
                printer = graphtage.printer.Printer(out_stream=stream, ansi_color=False)
                formatter.print(printer, tree)
                printer.flush(final=True)
                test_iter.send(stream.getvalue())
                new_obj = next(test_iter)
                self.assertEqual(orig_obj, new_obj)

        return test_func(self)
    return wrapper


class TestFormatting(TestCase):
    @staticmethod
    def make_random_int() -> int:
        return random.randint(-1000000, 1000000)

    @staticmethod
    def make_random_float() -> float:
        return random.random()

    @staticmethod
    def make_random_bool() -> bool:
        return random.choice([True, False])

    @staticmethod
    def make_random_str() -> str:
        return ''.join(random.choices(STR_BYTES, k=random.randint(0, 128)))

    @staticmethod
    def make_random_non_container():
        return random.choice([
            TestFormatting.make_random_int,
            TestFormatting.make_random_bool,
            TestFormatting.make_random_float,
            TestFormatting.make_random_str
        ])()

    @staticmethod
    def _make_random_obj(obj_stack):
        r = random.random()
        NON_CONTAINER_PROB = 0.1
        CONTAINER_PROB = (1.0 - NON_CONTAINER_PROB) / 2.0
        if r <= NON_CONTAINER_PROB:
            ret = TestFormatting.make_random_non_container()
        elif r <= NON_CONTAINER_PROB + CONTAINER_PROB:
            ret = []
            obj_stack.append(ret)
        else:
            ret = {}
            obj_stack.append(ret)
        return ret

    @staticmethod
    def make_random_obj(force_string_keys: bool = False):
        obj_stack = []
        ret = TestFormatting._make_random_obj(obj_stack)
        while obj_stack:
            expanding = obj_stack.pop()
            if isinstance(expanding, dict):
                for _ in range(int(random.betavariate(0.75, 5) * 10)):
                    if force_string_keys:
                        expanding[TestFormatting.make_random_str()] = TestFormatting._make_random_obj(obj_stack)
                    else:
                        expanding[TestFormatting.make_random_non_container()] = TestFormatting._make_random_obj(obj_stack)
            else:
                for _ in range(int(random.betavariate(0.75, 5) * 10)):
                    expanding.append(TestFormatting._make_random_obj(obj_stack))
        return ret

    def test_formatter_coverage(self):
        for name in graphtage.FILETYPES_BY_TYPENAME.keys():
            if not hasattr(self, f'test_{name}_formatting'):
                self.fail(f"Filetype {name} is missing a `test_{name}_formatting` test function")

    @filetype_test
    def test_json_formatting(self):
        orig_obj = TestFormatting.make_random_obj(force_string_keys=True)
        formatted_str = (yield orig_obj, json.dumps(orig_obj).encode('utf-8'))
        yield orig_obj, formatted_str
        try:
            yield json.loads(formatted_str)
        except json.decoder.JSONDecodeError as de:
            self.fail(f"""JSON decode error {de}: Original version:
        {orig_obj!r}
        Formatted version:
        {formatted_str!s}""")
