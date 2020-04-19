import io
import sys
from types import TracebackType
from typing import AnyStr, Iterable, Iterator, IO, List, Optional, TextIO, Type

from tqdm import tqdm, trange


class StatusWriter(IO[str]):
    def __init__(self, out_stream: Optional[TextIO] = None, quiet: bool = False):
        self.quiet = quiet
        if out_stream is None:
            out_stream = sys.stdout
        self.status_stream: TextIO = out_stream
        self._buffer: List[str] = []
        try:
            self.write_raw = self.quiet or (
                    out_stream.fileno() != sys.stderr.fileno() and out_stream.fileno() != sys.stdout.fileno()
            )
        except io.UnsupportedOperation:
            self.write_raw = False

    def tqdm(self, *args, **kwargs) -> tqdm:
        if self.quiet:
            kwargs['disable'] = True
        return tqdm(*args, **kwargs)

    def trange(self, *args, **kwargs) -> trange:
        if self.quiet:
            kwargs['disable'] = True
        return trange(*args, **kwargs)

    def flush(self, final=False):
        if final and self._buffer and not self._buffer[-1].endswith('\n'):
            self._buffer.append('\n')
        while self._buffer:
            if '\n' in self._buffer[0]:
                trailing_newline = self._buffer[0].endswith('\n')
                lines = self._buffer[0].split('\n')
                if not trailing_newline:
                    if len(self._buffer) == 1:
                        self._buffer.append(lines[-1])
                    else:
                        self._buffer[1] = f"{lines[-1]}{self._buffer[1]}"
                for line in lines[:-1]:
                    tqdm.write(line, file=self.status_stream)
                self._buffer = self._buffer[1:]
            elif len(self._buffer) == 1:
                break
            else:
                self._buffer = [''.join(self._buffer)]
        return self.status_stream.flush()

    def write(self, text: str) -> int:
        if self.write_raw:
            return self.status_stream.write(text)
        self._buffer.append(text)
        if '\n' in text:
            self.flush()
        return len(text)

    def close(self) -> None:
        self.flush(final=True)
        return self.status_stream.close()

    def fileno(self) -> int:
        return self.status_stream.fileno()

    def isatty(self) -> bool:
        return self.status_stream.isatty()

    def read(self, n: int = ...) -> AnyStr:
        return self.status_stream.read(n)

    def readable(self) -> bool:
        return self.status_stream.readable()

    def readline(self, limit: int = ...) -> AnyStr:
        return self.status_stream.readline(limit)

    def readlines(self, hint: int = ...) -> List[AnyStr]:
        return self.status_stream.readlines(hint)

    def seek(self, offset: int, whence: int = ...) -> int:
        return self.status_stream.seek(offset, whence)

    def seekable(self) -> bool:
        return self.status_stream.seekable()

    def tell(self) -> int:
        return self.status_stream.tell()

    def truncate(self, size: Optional[int] = ...) -> int:
        return self.status_stream.truncate(size)

    def writable(self) -> bool:
        return self.status_stream.writable()

    def writelines(self, lines: Iterable[AnyStr]) -> None:
        return self.status_stream.writelines(lines)

    def __next__(self) -> AnyStr:
        return next(self.status_stream)

    def __iter__(self) -> Iterator[AnyStr]:
        return iter(self.status_stream)

    def __enter__(self) -> IO[AnyStr]:
        return self.status_stream.__enter__()

    def __exit__(self, t: Optional[Type[BaseException]], value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        return self.status_stream.__exit__(t, value, traceback)

    def __delete__(self, instance):
        self.flush(final=True)
