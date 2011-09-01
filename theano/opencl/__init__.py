try:
    import pyopencl
    available = True
except:
    available = False

# test for opencl by doing something like:
"""
import theano.opencl
if theano.opencl.available:
     ...
"""

if available:
    import conf
    import type, var
    from basic_ops import host_from_cl, cl_from_host

