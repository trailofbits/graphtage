import sys
from typing import Optional, Union
from typing_extensions import Protocol

import colorama
from colorama import Back, Fore, Style
from colorama.ansi import AnsiFore, AnsiBack, AnsiStyle


class Writer(Protocol):
    def write(self, s: str) -> int:
        pass


class ANSIContext:
    def __init__(
            self,
            stream: Union[Writer, 'ANSIContext'],
            fore: Optional[AnsiFore] = None,
            back: Optional[AnsiBack] = None,
            style: Optional[AnsiStyle] = None,
    ):
        if isinstance(stream, ANSIContext):
            self.stream: Writer = stream.stream
            self._parent: Optional['ANSIContext'] = stream
        else:
            self.stream: Writer = stream
            self._parent: Optional['ANSIContext'] = None
        self._fore: Optional[AnsiFore] = fore
        self._back: Optional[AnsiBack] = back
        self._style: Optional[AnsiStyle] = style
        self.start_code: str = ''
        self.end_code: str = ''
        if self._fore is not None and (self._parent is None or self._fore != self.parent.fore):
            self.start_code += self._fore
            if self._parent is None or self.parent.fore is None:
                self.end_code = Fore.RESET
            else:
                self.end_code = self.parent.fore
        if self._back is not None and (self._parent is None or self._back != self.parent.back):
            self.start_code += self._back
            if self._parent is None or self.parent.back is None:
                self.end_code = Back.RESET
            else:
                self.end_code = self.parent.back
        if self._style is not None and (self._parent is None or self._style != self.parent.style):
            self.start_code += self._style
            if self._parent is None or self.parent.style is None:
                self.end_code = Style.RESET_ALL
            else:
                self.end_code = self.parent.style

    @property
    def fore(self) -> Optional[AnsiFore]:
        if self._fore is None and self._parent is not None:
            return self._parent.fore
        else:
            return self._fore

    @property
    def back(self) -> Optional[AnsiBack]:
        if self._back is None and self._parent is not None:
            return self._parent.back
        else:
            return self._back

    @property
    def style(self) -> Optional[AnsiStyle]:
        if self._style is None and self._parent is not None:
            return self._parent.style
        else:
            return self._style

    @property
    def parent(self) -> Optional['ANSIContext']:
        return self._parent

    def color(self, foreground_color: AnsiFore) -> 'ANSIContext':
        return ANSIContext(
            stream=self,
            fore=foreground_color
        )

    def background(self, bg_color: AnsiBack) -> 'ANSIContext':
        return ANSIContext(
            stream=self,
            back=bg_color
        )

    def bright(self) -> 'ANSIContext':
        return ANSIContext(
            stream=self,
            style=Style.BRIGHT
        )

    def dim(self) -> 'ANSIContext':
        return ANSIContext(
            stream=self,
            style=Style.DIM
        )

    def __enter__(self) -> Writer:
        self.stream.write(self.start_code)
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.write(self.end_code)


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


class Printer:
    def __init__(self, out_stream: Optional[Writer] = None, ansi_color: Optional[bool] = None):
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

    def color(self, foreground_color: AnsiFore) -> ANSIContext:
        if self.ansi_color:
            return ANSIContext(self, fore=foreground_color)
        else:
            return NullANSIContext()    # type: ignore

    def background(self, bg_color: AnsiBack) -> Optional[ANSIContext]:
        if self.ansi_color:
            return ANSIContext(self, back=bg_color)
        else:
            return NullANSIContext()    # type: ignore

    def bright(self) -> Optional[ANSIContext]:
        if self.ansi_color:
            return ANSIContext(self, style=Style.BRIGHT)
        else:
            return NullANSIContext()    # type: ignore

    def dim(self) -> Optional[ANSIContext]:
        if self.ansi_color:
            return ANSIContext(self, style=Style.DIM)
        else:
            return NullANSIContext()    # type: ignore

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
