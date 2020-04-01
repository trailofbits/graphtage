import sys
from collections import defaultdict
from typing import Dict, List, Optional, TextIO

import colorama
from colorama import Back, Fore, Style


class ANSIContext:
    def __init__(self, stream: TextIO, start_code: str, end_code: Optional[str] = None):
        self.start_code = start_code
        if end_code is None:
            end_code = ''
        self.end_code = end_code
        self.stream = stream

    def color(self, foreground_color: Fore):
        return ANSIContext(
            self.stream,
            self.start_code + foreground_color,
            Fore.RESET + self.end_code
        )

    def background(self, bg_color: Back):
        return ANSIContext(
            self.stream,
            self.start_code + bg_color,
            Back.RESET + self.end_code
        )

    def bright(self):
        return ANSIContext(
            self.stream,
            self.start_code + Style.BRIGHT,
            Style.RESET_ALL + self.end_code
        )

    def dim(self):
        return ANSIContext(
            self.stream,
            self.start_code + Style.DIM,
            Style.RESET_ALL + self.end_code
        )

    def __enter__(self):
        self.stream.write(self.start_code)
        ANSI_CONTEXT_STACKS[self.stream].append(self)
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        stack = ANSI_CONTEXT_STACKS[self.stream]
        assert stack[-1] == self
        if self.end_code is not None and self.end_code:
            self.stream.write(self.end_code)
        stack.pop()
        for context in stack:
            self.stream.write(context.start_code)


class NullANSIContext:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getattr__(self, item):
        def fake_fun(*args, **kwargs):
            return self

        return fake_fun


ANSI_CONTEXT_STACKS: Dict[TextIO, List[ANSIContext]] = defaultdict(list)


class Printer:
    def __init__(self, out_stream: Optional[TextIO] = None, ansi_color: Optional[bool] = None):
        if out_stream is None:
            out_stream = sys.stdout
        self.out_stream = out_stream
        self.indents = 0
        if ansi_color is None:
            self.ansi_color = out_stream.isatty()
        else:
            self.ansi_color = ansi_color
        if self.ansi_color:
            colorama.init()

    def write(self, s: str):
        self.out_stream.write(s)

    def newline(self):
        self.write('\n')
        self.write(' ' * (4 * self.indents))

    def color(self, foreground_color: Fore):
        if self.ansi_color:
            return ANSIContext(self, foreground_color, Fore.RESET)
        else:
            return NullANSIContext()

    def background(self, bg_color: Back) -> Optional[ANSIContext]:
        if self.ansi_color:
            return ANSIContext(self, bg_color, Back.RESET)
        else:
            return NullANSIContext()

    def bright(self) -> Optional[ANSIContext]:
        if self.ansi_color:
            return ANSIContext(self, Style.BRIGHT, Style.RESET_ALL)
        else:
            return NullANSIContext()

    def dim(self) -> Optional[ANSIContext]:
        if self.ansi_color:
            return ANSIContext(self, Style.DIM, Style.RESET_ALL)
        else:
            return NullANSIContext()

    def indent(self):
        class Indent:
            def __init__(self, printer):
                self.printer = printer

            def __enter__(self):
                self.printer.indents += 1
                return self.printer

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.printer.indents -= 1

        return Indent(self)


DEFAULT_PRINTER: Printer = Printer()