import _thread
import threading
from contextlib import contextmanager


@contextmanager
def run_with_time_limit(seconds: int):
    """Runs the wrapped block, raising :class:`TimeoutError` if it takes longer than `seconds`.

    The timer interrupts the main thread, so the block must be interruptible Python code, and this context manager
    must be entered from the main thread.

    Any other exception the block raises propagates unchanged.

    Args:
        seconds: The wall-clock budget for the block.

    Raises:
        TimeoutError: If the block is still running when the budget expires.

    """
    timed_out = False

    def interrupt():
        nonlocal timed_out
        timed_out = True
        _thread.interrupt_main()

    timer = threading.Timer(seconds, interrupt)
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        if not timed_out:
            raise
        raise TimeoutError(f"timeout after {seconds} seconds")
    finally:
        timer.cancel()
