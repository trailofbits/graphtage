import threading
import _thread
from contextlib import contextmanager


@contextmanager
def run_with_time_limit(seconds: int):
    timer = threading.Timer(seconds, _thread.interrupt_main)
    timer.start()

    try:
        yield
        return
    except:
        pass
    finally:
        timer.cancel()
    raise TimeoutError(f"timeout after {seconds} seconds")
