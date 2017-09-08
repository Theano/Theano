from __future__ import absolute_import, print_function, division

"""
Helper routines for generating gpu kernels for nvcc.

"""

try:
    from pygpu import gpuarray
except ImportError:
    pass


def nvcc_kernel(name, params, body):
    """
    Return the c code of a kernel function.

    Parameters
    ----------
    params
        The parameters to the function as one or more strings.
    body
        The [nested] list of statements for the body of the function.
        These will be separated by ';' characters.

    """
    paramstr = ', '.join(params)

    def flatbody():
        for b in body:
            if isinstance(b, (list, tuple)):
                for bb in b:
                    yield bb
            else:
                yield b
    bodystr = ';\n'.join(flatbody())
    return """#include "cluda.h"

    KERNEL void %(name)s (%(paramstr)s)
    {
        %(bodystr)s;
    }
    """ % locals()


def code_version(version):
    """
    Decorator to support version-based cache mechanism.

    """
    if not isinstance(version, tuple):
        raise TypeError('version must be tuple', version)

    def deco(f):
        f.code_version = version
        return f
    return deco

UNVERSIONED = ()


@code_version((2,))
def inline_reduce(N, buf, pos, count, manner_fn):
    """
    Return C++ code for a function that reduces a contiguous buffer.

    Parameters
    ----------
    N
        Length of the buffer.
    buf
        buffer pointer.
    pos
        Index of executing thread.
    count
        Number of executing threads.
    manner_fn
        A function that accepts strings of arguments a and b, and
        returns c code for their reduction.

          return "%(a)s + %(b)s"

        for a sum reduction.

    Notes
    -----
    `buf` should be in gpu shared memory, we access it many times.

    This function leaves the answer in position 0 of the buffer. The
    rest of the buffer is trashed by this function.

    """
    loop_line = manner_fn("%s[%s]" % (buf, pos), "%s[i]" % (buf))
    r_n = manner_fn("%s[%s]" % (buf, pos), "%s[%s+_n]" % (buf, pos))

    return """
    {
        // This function trashes buf[1..warpSize],
        // leaving the reduction result in buf[0].

        if (%(pos)s < warpSize) {
            for (int i = %(pos)s + warpSize; i < %(N)s; i += warpSize)
            {
                %(buf)s[%(pos)s] = %(loop_line)s;
            }
        }
        __syncthreads();
        //reduce so that %(pos)s 0 has the reduction of everything
        for (unsigned int _n = warpSize / 2; _n > 0; _n /= 2) {
          if (%(pos)s < _n && %(pos)s + _n < %(N)s)
            %(buf)s[%(pos)s] = %(r_n)s;
          __syncthreads();
        }
    }
    """ % locals()


@code_version(inline_reduce.code_version)
def inline_reduce_max(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count,
                         lambda a, b: "max(%s, %s)" % (a, b))


@code_version(inline_reduce.code_version)
def inline_reduce_sum(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count,
                         lambda a, b: "%s + %s" % (a, b))


@code_version(inline_reduce.code_version)
def inline_reduce_min(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count,
                         lambda a, b: "min(%s, %s)" % (a, b))


@code_version(inline_reduce.code_version)
def inline_reduce_prod(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count,
                         lambda a, b: "%s * %s" % (a, b))


@code_version((2,) + inline_reduce_max.code_version +
              inline_reduce_sum.code_version)
def inline_softmax(N, buf, buf2, threadPos, threadCount, dtype="float32"):
    """
    Generate code for a softmax.

    On entry, `buf` and `buf2` must contain two identical copies of
    the input to softmax.

    After the code returns `buf` contains the softmax, `buf2` contains
    un-normalized softmax.

    Parameters
    ----------
    N
        Length of the buffer.
    threadPos
        Index of executing thread.
    threadCount
        Number of executing threads.
    dtype
        Dtype of the softmax's output.

    Notes
    -----
    `buf` and `buf2` should be in gpu shared memory, we access it many
    times.

    We use __i as an int variable in a loop.

    """
    ctype = gpuarray.dtype_to_ctype(dtype)
    # get max of buf (trashing all but buf[0])
    return [inline_reduce_max(N, buf, threadPos, threadCount),
            '__syncthreads()',
            ('%s row_max = ' + buf + '[0]') % ctype,
            '__syncthreads()',
            'for(int __i=' + threadPos + '; __i<' + N +
            '; __i+=' + threadCount + '){',
            buf + '[__i] = exp(' + buf2 + '[__i] - row_max)',
            buf2 + '[__i] = ' + buf + '[__i]',
            '}',
            '__syncthreads()',
            inline_reduce_sum(N, buf, threadPos, threadCount),
            '__syncthreads()',
            ('%s row_sum = ' + buf + '[0]') % ctype,
            '__syncthreads()',
            # divide each exp() result by the sum to complete the job.
            'for(int __i=' + threadPos + '; __i<' + N +
            '; __i+=' + threadCount + '){',
            buf + '[__i] = ' + buf2 + '[__i] / row_sum',
            '}',
            '__syncthreads()',
            ]


@code_version((3,))
def inline_reduce_fixed_shared(N, buf, x, stride_x, load_x, pos, count,
                               manner_fn, manner_init,
                               b='', stride_b='', load_b='', dtype='float32'):
    """
    Return C++ code for a function that reduces a contiguous buffer.

    This function leaves the answer in position 0 of the buffer. The
    rest of the buffer is trashed by this function.

    Parameters
    ----------
    N
        Length of the buffer.
    buf
        Buffer pointer of size warpSize * sizeof(dtype).
    x
        Input data.
    stride_x
        Input data stride.
    load_x
        Wrapper to read from x.
    pos
        Index of executing thread.
    count
        Number of executing threads.
    manner_fn
        A function that accepts strings of arguments a and b, and
        returns c code for their reduction.

          return "%(a)s + %(b)s"

        for a sum reduction.
    manner_init
        A function that accepts strings of arguments a and return c
        code for its initialization.
    b
        Optional, pointer to the bias.
    stride_b
        Optional, the stride of b if b is provided.
    load_b
        Optional, wrapper to read from b if b is provided.
    dtype
        Optional, the dtype of the output.

    Notes
    -----
    `buf` should be in gpu shared memory, we access it many times.

    """
    if b:
        init = manner_init("%(load_x)s(%(x)s[%(pos)s * %(stride_x)s]) +"
                           " %(load_b)s(%(b)s[%(pos)s * %(stride_b)s])" % locals())
        loop_line = manner_fn("red",
                              manner_init("%(load_x)s(%(x)s[i * %(stride_x)s]) + "
                                          "%(load_b)s(%(b)s[i * %(stride_b)s])" %
                                          locals()))
    else:
        init = manner_init("%(load_x)s(%(x)s[%(pos)s * %(stride_x)s])" % locals())
        loop_line = manner_fn("red", manner_init("%(load_x)s(%(x)s[i * %(stride_x)s])" %
                                                 locals()))
    loop_line2 = manner_fn("%s[%s]" % (buf, pos),
                           "%s[i]" % buf)
    r_n = manner_fn("%s[%s]" % (buf, pos), "%s[%s+_n]" % (buf, pos))

    ctype = gpuarray.dtype_to_ctype(dtype)
    return """
    {
        // This function trashes buf[1..n_threads],
        // leaving the reduction result in buf[0].
        %(ctype)s red = %(init)s;
        #pragma unroll 16
        for (int i = %(pos)s + %(count)s; i<%(N)s; i += %(count)s) {
          red = %(loop_line)s;
        }
        buf[%(pos)s] = red;
        __syncthreads();
        if (%(pos)s < warpSize) {
            for (int i = %(pos)s + warpSize; i < %(count)s; i += warpSize) {
                %(buf)s[%(pos)s] = %(loop_line2)s;
            }
        }
        __syncthreads();
        //reduce so that %(pos)s 0 has the reduction of everything
        for (unsigned int _n = warpSize / 2; _n > 0; _n /= 2) {
          if (%(pos)s < _n && %(pos)s + _n < %(N)s)
            %(buf)s[%(pos)s] = %(r_n)s;
          __syncthreads();
        }
    }
    """ % locals()


@code_version(inline_reduce_fixed_shared.code_version)
def inline_reduce_fixed_shared_max(N, buf, x, stride_x, load_x, pos, count,
                                   b='', stride_b='', load_b='',
                                   dtype='float32'):
    return inline_reduce_fixed_shared(N, buf, x, stride_x, load_x, pos, count,
                                      lambda a, b: "max(%s, %s)" % (a, b),
                                      lambda a: a,
                                      b, stride_b, load_b, dtype)


@code_version((2,) + inline_reduce_max.code_version +
              inline_reduce_sum.code_version)
def inline_softmax_fixed_shared(N, buf, x, stride_x, load_x,
                                sm, sm_stride, write_sm,
                                threadPos, threadCount,
                                b='', stride_b='', load_b='',
                                dtype="float32"):
    """
    Generate code to perform softmax with a fixed amount of shared
    memory.

    On entry, `buf` is assumed to be empty.

    On exit, `buf[0]` contains the softmax, `buf2` contains
    un-normalized softmax.

    Parameters
    ----------
    N
        Length of the buffer, atleast waprSize(32).
    buf
        A shared memory buffer of size warpSize * sizeof(dtype).
    x
        A ptr to the gpu memory where the row is stored.
    stride_x
        The stride between each element in x.
    load_x
        Wrapper to read from x.
    sm
        A ptr to the gpu memory to store the result.
    sm_stride
        The stride between each sm element.
    write_sm
        Wrapper before writing to sm.
    threadPos
        Index of executing thread.
    threadCount
        Number of executing threads.
    b
        Optional, pointer to the bias.
    stride_b
        Optional, the stride of b if b is provided.
    load_b
        Optional, wrapper to read from b if b is provided.
    dtype
        Optional, the dtype of the softmax's output if not float32.

    Notes
    -----
    `buf` should be in gpu shared memory, we access it many times.

    We use tx as an int variable in a loop.

    """
    ctype = gpuarray.dtype_to_ctype(dtype)
    ret = [
        # get max of buf (trashing all but buf[0])
        inline_reduce_fixed_shared_max(N, buf, x, stride_x, load_x,
                                       threadPos, threadCount,
                                       b, stride_b, load_b,
                                       dtype),
        '__syncthreads()',
        ('%s row_max = ' + buf + '[0]') % ctype,
        '__syncthreads()',
        inline_reduce_fixed_shared(N, buf, x, stride_x, load_x,
                                   threadPos, threadCount,
                                   lambda a, b: "%s + %s" % (a, b),
                                   lambda a: "exp(%s - row_max)" % a,
                                   b, stride_b, load_b, dtype),
        '__syncthreads()',
        ('%s row_sum = ' + buf + '[0]') % ctype,
        '__syncthreads()',
        "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
        ]
    # This set all value correctly
    if b:
        ret += [
            "%(sm)s[tx * %(sm_stride)s] = "
            "  %(write_sm)s(exp(%(load_x)s(%(x)s[tx * %(stride_x)s]) +"
            "            %(load_b)s(%(b)s[tx * %(stride_b)s]) - row_max)"
            " / row_sum)" % locals()]
    else:
        ret += [
            "%(sm)s[tx * %(sm_stride)s] = "
            "%(write_sm)s(exp(%(load_x)s(%(x)s[tx * %(stride_x)s]) - row_max)"
            " / row_sum)" % locals()]
    ret += [
        "}",
        '__syncthreads()',
    ]
    return ret
