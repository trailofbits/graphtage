import json
import random
from io import StringIO
from unittest import TestCase

from tqdm import trange

import graphtage

STR_BYTES: str = ''.join([
    chr(i) for i in range(32, 127)
] + ['\n', '\t', '\r'])


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
    def make_random_obj():
        obj_stack = []
        ret = TestFormatting._make_random_obj(obj_stack)
        while obj_stack:
            expanding = obj_stack.pop()
            if isinstance(expanding, dict):
                for _ in range(int(random.betavariate(0.75, 5) * 10)):
                    expanding[TestFormatting.make_random_non_container()] = TestFormatting._make_random_obj(obj_stack)
            else:
                for _ in range(int(random.betavariate(0.75, 5) * 10)):
                    expanding.append(TestFormatting._make_random_obj(obj_stack))
        return ret

    def test_formatter_coverage(self):
        for name in graphtage.FILETYPES_BY_TYPENAME.keys():
            if not hasattr(self, f'test_{name}_formatting'):
                self.fail(f"Filetype {name} is missing a `test_{name}_formatting` test function")

    def test_json_formatting(self):
        filetype = graphtage.FILETYPES_BY_TYPENAME['json']
        formatter = filetype.get_default_formatter()
        for _ in trange(1000):
            orig_obj = TestFormatting.make_random_obj()
            with graphtage.utils.Tempfile(json.dumps(orig_obj).encode('utf-8')) as t:
                tree = filetype.build_tree(t)
                stream = StringIO()
                printer = graphtage.printer.Printer(out_stream=stream, ansi_color=False)
                formatter.print(printer, tree)
                printer.flush(final=True)
                # Now confirm the formatted output is still valid JSON:
                try:
                    new_obj = json.loads(stream.getvalue())
                except json.decoder.JSONDecodeError as de:
                    self.fail(f"""JSON decode error {de}: Original version:
{orig_obj!r}
Formatted version:
{stream.getvalue()!s}""")
                self.assertEqual(orig_obj, new_obj)
