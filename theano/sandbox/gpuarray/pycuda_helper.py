try:
    from pycuda.driver import Context
    if not hasattr(Context, 'attach'):
        raise ImportError('too old')
except ImportError:
    Context = None

pycuda_initialized = False
pycuda_context = None


def ensure_pycuda_context():
    global pycuda_context, pycuda_initialized
    if not pycuda_initialized:
        if Context is None:
            raise RuntimeError("PyCUDA not found or too old.")
        else:
            pycuda_context = Context.attach()
            import atexit
            atexit.register(pycuda_context.detach)
            pycuda_initialized = True
    return pycuda_context
