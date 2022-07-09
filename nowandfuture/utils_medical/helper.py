from logging import warning


def check_memoryleak(function_, *args, **kwargs):
    import tracemalloc, gc
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    print("Before: memory usage is {}MB, peak was {}".format(current / 1e6, peak / 1e6))
    function_(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    print("After(GC Prepared): memory usage is {}MB, peak was {}MB".format(current / 1e6, peak / 1e6))
    gc.collect()
    after, peak = tracemalloc.get_traced_memory()
    print("After(GC Done): memory usage is {}MB, peak was {}MB".format(after / 1e6, peak / 1e6))
    tracemalloc.stop()
    if (current - after - 0.09) / after < 0.1:
        warning("This function may be memory leak")