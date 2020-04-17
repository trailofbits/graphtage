import sys
from typing import List, Optional, TextIO

from tqdm import tqdm, trange


class StatusWriter:
    def __init__(self, out_stream: Optional[TextIO] = None, quiet: bool = False):
        self.quiet = quiet
        if out_stream is None:
            out_stream = sys.stderr
        self.status_stream: TextIO = out_stream
        self._buffer: List[str] = []

    def tqdm(self, *args, **kwargs) -> tqdm:
        if self.quiet:
            kwargs['disable'] = True
        return tqdm(*args, **kwargs)

    def trange(self, *args, **kwargs) -> trange:
        if self.quiet:
            kwargs['disable'] = True
        return trange(*args, **kwargs)

    def flush(self):
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
        if self.quiet:
            return self.status_stream.write(text)
        self._buffer.append(text)
        if '\n' in text:
            self.flush()
        return len(text)
