"""A module for printing status messages and progress bars to the command line."""

import io
import sys
from types import TracebackType
from typing import AnyStr, Iterable, Iterator, IO, List, Optional, TextIO, Type

from tqdm import tqdm, trange


class StatusWriter(IO[str]):
    """A writer compatible with the :class:`graphtage.printer.Writer` protocol that can print status.

    See :meth:`StatusWriter.tqdm` and :meth:`StatusWriter.trange`. If :attr:`StatusWriter.status_stream` is either
    :attr:`sys.stdout` or :attr:`sys.stderr`, then bytes printed to this writer will be buffered. For each full line
    buffered, a call to :func:`tqdm.write` will be made.

    A status writer whose lifetime is not controlled by instantiation in a ``with`` block must be manually flushed
    with :meth:`StatusWriter.flush(final=True)<StatusWriter.flush>` after its final write, or else the last line
    written may be lost.

    """
    def __init__(self, out_stream: Optional[TextIO] = None, quiet: bool = False):
        """Initializes a status writer.

        Args:
            out_stream: An optional stream to which to write. If omitted this defaults to :attr:`sys.stdout`.
            quiet: Whether or not :mod:`tqdm` status messages and progress should be suppressed.

        """
        self.quiet = quiet
        """Whether or not :mod:`tqdm` status messages and progress should be suppressed."""
        self._reentries: int = 0
        if out_stream is None:
            out_stream = sys.stdout
        self.status_stream: TextIO = out_stream
        """The status stream to which to print."""
        self._buffer: List[str] = []
        try:
            self.write_raw = self.quiet or (
                    out_stream.fileno() != sys.stderr.fileno() and out_stream.fileno() != sys.stdout.fileno()
            )
            """If :const:`True`, this writer *will not* buffer output and use :func:`tqdm.write`.
            
            This defaults to::
            
                self.write_raw = self.quiet or (
                    out_stream.fileno() != sys.stderr.fileno() and out_stream.fileno() != sys.stdout.fileno()
                )
            
            """
        except io.UnsupportedOperation as e:
            self.write_raw = True

    def tqdm(self, *args, **kwargs) -> tqdm:
        """Returns a :class:`tqdm.tqdm` object."""
        if self.quiet:
            kwargs['disable'] = True
        return tqdm(*args, **kwargs)

    def trange(self, *args, **kwargs) -> trange:
        """Returns a :class:`tqdm.trange` object."""
        if self.quiet:
            kwargs['disable'] = True
        return trange(*args, **kwargs)

    def flush(self, final=False):
        """Flushes this writer.

        If :obj:`final` is :const:`True`, any extra bytes will be flushed along with a final newline.

        """
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

    @property
    def closed(self) -> bool:
        return self.status_stream.closed

    @property
    def mode(self) -> str:
        return self.status_stream.mode

    @property
    def name(self) -> str:
        return self.status_stream.name

    def __next__(self) -> AnyStr:
        return next(self.status_stream)

    def __iter__(self) -> Iterator[AnyStr]:
        return iter(self.status_stream)

    def __enter__(self) -> IO[AnyStr]:
        self._reentries += 1
        return self

    def __exit__(self, t: Optional[Type[BaseException]], value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> Optional[bool]:
        self._reentries -= 1
        if self._reentries == 0:
            self.flush(final=True)

    def __delete__(self, instance):
        self.flush(final=True)
