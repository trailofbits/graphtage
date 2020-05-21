import sys
from abc import ABC
from collections.abc import Mapping as AbstractMapping
from functools import partial, wraps
from multiprocessing import cpu_count
from multiprocessing import Pool as MPool
from typing import Any, Dict, Optional, TypeVar
from typing_extensions import Protocol


if sys.version_info[0] == 3 and sys.version_info[1] < 8:
    from multiprocessing import current_process

    def is_main_process() -> bool:
        return current_process().name == 'MainProcess'

else:
    # Python 3.8 and newer
    from multiprocessing import parent_process

    def is_main_process():
        return parent_process() is None


class EmptyDict(AbstractMapping, Dict):
    def __getitem__(self, k):
        raise KeyError(k)

    def __len__(self) -> int:
        return 0

    def __iter__(self):
        return iter(())

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return "{}"


class Pool(Protocol):
    def apply(self, func, args=(), kwds: Dict[str, Any] = EmptyDict()):
        raise NotImplementedError()

    def map(self, func, iterable, chunksize: Optional[int] = None):
        raise NotImplementedError()

    def imap(self, func, iterable, chunksize: Optional[int] = None):
        raise NotImplementedError()

    def imap_unordered(self, func, iterable, chunksize: Optional[int] = None):
        raise NotImplementedError()

    def starmap(self, func, iterable, chunksize: Optional[int] = None):
        raise NotImplementedError()

    def num_workers(self) -> int:
        raise NotImplementedError()

    def is_closed(self) -> bool:
        raise NotImplementedError()

    def __enter__(self) -> 'Pool':
        raise NotImplementedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError()


P = TypeVar('P', bound=Pool)


class AbstractPool(Pool, ABC):
    def __init__(self, num_workers: int):
        self._num_workers = num_workers
        self.context_count: int = 0

    def num_workers(self) -> int:
        return self._num_workers

    def __enter__(self: P) -> P:
        self.context_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context_count -= 1


class SameProcessPool(AbstractPool):
    def __init__(self, num_workers: Optional[int] = 0):
        if num_workers > 1:
            raise ValueError(f"SameProcessPool can have at most one worker (the same process).")
        super().__init__(0)

    def apply(self, func, args=(), kwds: Dict[str, Any] = EmptyDict()):
        return func(*args, **kwds)

    def map(self, func, iterable, chunksize: Optional[int] = None):
        return map(func, iterable)

    imap = map

    imap_unordered = map

    def starmap(self, func, iterable, chunksize: Optional[int] = None):
        return [func(*args) for args in iterable]

    def is_closed(self) -> bool:
        return False


class MultiProcessPool(AbstractPool):
    def __init__(self, num_workers: Optional[int] = None):
        if num_workers is None:
            num_workers = cpu_count()
        self._pool = MPool(num_workers)
        self._closed: bool = False
        super().__init__(num_workers)

    def apply(self, *args, **kwargs):
        return self._pool.apply(*args, **kwargs)

    def map(self, *args, **kwargs):
        return self._pool.map(*args, **kwargs)

    def imap(self, *args, **kwargs):
        return self._pool.imap(*args, **kwargs)

    def imap_unordered(self, *args, **kwargs):
        return self._pool.imap_unordered(*args, **kwargs)

    def starmap(self, *args, **kwargs):
        return self._pool.starmap(*args, **kwargs)

    def __enter__(self) -> 'MultiProcessPool':
        if self.context_count == 0:
            self._pool.__enter__()
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if self.context_count == 0:
            self._pool.__exit__(exc_type, exc_val, exc_tb)
            self._closed = True

    def is_closed(self) -> bool:
        return self._closed


_DEFAULT_POOL: Optional[Pool] = None


def make_pool(processes: int):
    if processes > 1:
        return MultiProcessPool(processes)
    else:
        return SameProcessPool()


def default_pool() -> Pool:
    if not is_main_process():
        return SameProcessPool()

    global _DEFAULT_POOL

    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = make_pool(cpu_count())
    elif _DEFAULT_POOL.is_closed():
        _DEFAULT_POOL = _DEFAULT_POOL.__class__(_DEFAULT_POOL.num_workers())
    return _DEFAULT_POOL


def set_default_pool(pool: Pool):
    global _DEFAULT_POOL

    _DEFAULT_POOL = pool


def parallelize(func=None, *, parallel_func=None):
    if func is None:
        return partial(parallelize, parallel_func=parallel_func)

    @wraps(func)
    def wrapper(*args, pool: Optional[Pool] = None, **kwargs):
        is_main = is_main_process()

        if not is_main and parallel_func is not None:
            return func(*args, **kwargs)

        if pool is None and is_main:
            pool = default_pool()

        if not is_main or isinstance(pool, SameProcessPool):
            if parallel_func is not None:
                return func(*args, **kwargs)
            else:
                return func(*args, **kwargs, pool=pool)

        with pool as p:
            if parallel_func is not None:
                kwargs = kwargs.copy()
                kwargs['pool'] = p
                return parallel_func(*args, **kwargs)
            else:
                return p.apply(func, args, kwargs)

    return wrapper
