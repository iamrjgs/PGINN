import time
import math
import numpy as np
import cupy as cp
from numba import cuda, float32
from functools import wraps

def timed_method(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            cuda.synchronize()
        except:
            pass
        start = time.perf_counter()
        result = fn(self, *args, **kwargs)
        try:
            cuda.synchronize()
        except:
            pass
        end = time.perf_counter()
        self.timings[fn.__name__] = end - start
        return result
    return wrapper

class TimeProfiler:
    def __init__(self, prefix, on=True):
        self.last = time.perf_counter()
        self.prefix = prefix
        self.on = on

    def mark(self, label=""):
        if self.on:
            cuda.synchronize()

            now = time.perf_counter()
            elapsed = now - self.last
            print(f"{self.prefix} | {label} elapsed: {elapsed:.6f} sec")
            self.last = now