import sys
from collections import defaultdict
from typing import Dict, List, Optional, Union
from typing_extensions import Protocol

import colorama
from colorama import Back, Fore, Style
from colorama.ansi import AnsiFore, AnsiBack, AnsiStyle


class Writer(Protocol):
    def write(self, s: str) -> int:
        pass


class StrikethroughWriter(Writer):
    def __init__(self, parent: Writer):
        self.parent = parent

    def write(self, s: str) -> int:
        return self.parent.write("".join(f"{c}\u0336" for c in s))


class ANSIContext:
    def __init__(
            self,
            stream: Union[Writer, 'ANSIContext'],
            fore: Optional[AnsiFore] = None,
            back: Optional[AnsiBack] = None,
            style: Optional[AnsiStyle] = None,
            strikethrough: Optional[bool] = None
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
        self._start_code: Optional[str] = None
        self._end_code: Optional[str] = None
        self._strikethrough: Optional[bool] = strikethrough
        self._strikethrough_before: bool = False
        self.is_applied: bool = False

    @property
    def start_code(self) -> str:
        if self._start_code is None:
            self._set_codes()
        return self._start_code

    @property
    def end_code(self) -> str:
        if self._end_code is None:
            self._set_codes()
        return self._end_code

    def _set_codes(self):
        if not self.is_applied:
            raise ValueError("start_code and end_code should only be called after an ANSIContext has been __enter__'d")

        self._start_code: str = ''
        self._end_code: str = ''

        contexts = ANSI_CONTEXT_STACK[self.stream]
        if contexts:
            if self._parent is None:
                self._parent: Optional['ANSIContext'] = contexts[-1]
            else:
                if not self.root.is_applied:
                    self.root._parent = contexts[-1]

        parent_end_code: str = ''
        if self.parent is not None and not self.parent.is_applied:
            self.parent.is_applied = True
            self._start_code += self.parent.start_code
            parent_end_code = self.parent.end_code

        if self._fore is not None and (self._parent is None or self._fore != self.parent.fore):
            self._start_code += self._fore
            if self._parent is None or self.parent.fore is None:
                self._end_code = Fore.RESET
            else:
                self._end_code = self.parent.fore
        if self._back is not None and (self._parent is None or self._back != self.parent.back):
            self._start_code += self._back
            if self._parent is None or self.parent.back is None:
                self._end_code = Back.RESET
            else:
                self._end_code = self.parent.back
        if self._style is not None and (self._parent is None or self._style != self.parent.style):
            self._start_code += self._style
            if self._parent is None or self.parent.style is None:
                self._end_code = Style.RESET_ALL
            else:
                self._end_code = self.parent.style

        self._end_code += parent_end_code

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
    def strikethrough(self) -> bool:
        if self._strikethrough is None:
            if self._parent is not None:
                return self._parent.strikethrough
            else:
                return False
        else:
            return self._strikethrough

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

    def strike(self) -> 'ANSIContext':
        if not isinstance(self.stream, Printer):
            raise ValueError("Strikethrough is currently only supported with writers of type `Printer`")
        return ANSIContext(stream=self, strikethrough=True)

    @property
    def root(self):
        root = self
        while root._parent is not None:
            root = root._parent
        return root

    def _disable_strikethrough(self):
        if isinstance(self.stream, Printer):
            self.stream.strikethrough = False

    def _reenable_strikethrough(self):
        if isinstance(self.stream, Printer):
            self.stream.strikethrough = self.strikethrough

    def __enter__(self) -> Writer:
        assert not self.is_applied
        self.is_applied = True
        if isinstance(self.stream, Printer):
            self._strikethrough_before = self.stream.strikethrough
        self._disable_strikethrough()
        self.stream.write(self.start_code)
        self._reenable_strikethrough()
        ANSI_CONTEXT_STACK[self.stream].append(self)
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.is_applied
        self._disable_strikethrough()
        self.stream.write(self.end_code)
        if isinstance(self.stream, Printer):
            self.stream.strikethrough = self._strikethrough_before
        self.is_applied = False
        self._start_code = None
        self._end_code = None
        assert ANSI_CONTEXT_STACK[self.stream] and ANSI_CONTEXT_STACK[self.stream][-1] == self
        ANSI_CONTEXT_STACK[self.stream].pop()


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


ANSI_CONTEXT_STACK: Dict[Writer, List[ANSIContext]] = defaultdict(list)


class Printer:
    def __init__(self, out_stream: Optional[Writer] = None, ansi_color: Optional[bool] = None):
        if out_stream is None:
            out_stream = sys.stdout
        self.out_stream: Writer = out_stream
        self.indents = 0
        if ansi_color is None:
            self.ansi_color = out_stream.isatty()
        else:
            self.ansi_color = ansi_color
        if self.ansi_color:
            colorama.init()
        self._strikethrough = False

    @property
    def strikethrough(self) -> bool:
        return self._strikethrough

    @strikethrough.setter
    def strikethrough(self, value: bool):
        if self._strikethrough and not value:
            assert isinstance(self.out_stream, StrikethroughWriter)
            self.out_stream = self.out_stream.parent
            self._strikethrough = value
        elif not self._strikethrough and value:
            assert not isinstance(self.out_stream, StrikethroughWriter)
            self.out_stream = StrikethroughWriter(self.out_stream)
            self._strikethrough = value

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

    def background(self, bg_color: AnsiBack) -> ANSIContext:
        if self.ansi_color:
            return ANSIContext(self, back=bg_color)
        else:
            return NullANSIContext()    # type: ignore

    def bright(self) -> ANSIContext:
        if self.ansi_color:
            return ANSIContext(self, style=Style.BRIGHT)
        else:
            return NullANSIContext()    # type: ignore

    def dim(self) -> ANSIContext:
        if self.ansi_color:
            return ANSIContext(self, style=Style.DIM)
        else:
            return NullANSIContext()    # type: ignore

    def strike(self) -> ANSIContext:
        if self.ansi_color:
            return ANSIContext(self, strikethrough=True)
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
