import pyopencl as cl
import pyopencl.array
import numpy

from theano.configparser import config, AddConfigVar, StrParam

AddConfigVar('opencl.devices',
             """Devices to use to create the OpenCL context.
             Can be set to one of "ACCELERATOR", "ALL", "CPU", "DEFAULT", 
             "GPU" or to a comma separeted list of devices to use.
             XXX: DOES NOT DO ANYTHING RIGHT NOW
             """,
             StrParam("DEFAULT"))

ctx = cl.create_some_context(interactive=False)
default_queue = cl.CommandQueue(ctx)

def filter_dtype(dtype):
    if dtype != 'float32':
        raise TypeError('I am lazy')
    # CPU contexts support all types (except complex{64,128})
    # GPU contexts generally support all int types + float32 and break non-obviously with float64 (the sneaky bastards)

def filter_bcast(obj, broadcast):
    if len(obj.shape) != len(broadcast):
        raise ValueError("Wrong rank")
    for d, b in zip(obj.shape, broadcast):
        if b and d != 1:
            raise ValueError("Non-unit size in broacastable dimension")

def filter_array(obj, broadcast):
    if not isinstance(obj, cl.array.Array):
        raise TypeError("Not an OpenCL Array")
    filter_dtype(obj.dtype)
    filter_broadcast(obj, broadcast)
