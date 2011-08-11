""" Helper routines for generating gpu kernels for nvcc.
"""
def nvcc_kernel(name, params, body):
    """Return the c code of a kernel function.

    :param params: the parameters to the function as one or more strings

    :param body: the [nested] list of statements for the body of the function.  These will be
    separated by ';' characters.
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
    return """__global__ void %(name)s (%(paramstr)s)
    {
        %(bodystr)s;
    }
    """ %locals()

def code_version(version):
    """decorator to support version-based cache mechanism"""
    if not isinstance(version, tuple):
        raise TypeError('version must be tuple', version)
    def deco(f):
        f.code_version = version
        return f
    return deco

UNVERSIONED = ()

@code_version((1,))
def inline_reduce(N, buf, pos, count, manner_fn):
    """
    Return C++ code for a function that reduces a contiguous buffer.

    :param N: length of the buffer
    :param buf: buffer pointer
    :param pos: index of executing thread
    :param count: number of executing threads
    :param manner_fn: a function that accepts strings of arguments a and b, and returns c code
    for their reduction. (Example: return "%(a)s + %(b)s" for a sum reduction).

    :postcondition:
    This function leaves the answer in position 0 of the buffer.  The rest of the buffer is
    trashed by this function.

    :note: buf should be in gpu shared memory, we access it many times.

    """
    loop_line = manner_fn("%s[%s]"%(buf,pos), "%s[i]" %(buf))
    r_16 = manner_fn("%s[%s]" %(buf, pos), "%s[%s+16]" %(buf, pos))
    r_8 = manner_fn("%s[%s]" %(buf, pos), "%s[%s+8]" %(buf, pos))
    r_4 = manner_fn("%s[%s]" %(buf, pos), "%s[%s+4]" %(buf, pos))
    r_2 = manner_fn("%s[%s]" %(buf, pos), "%s[%s+2]" %(buf, pos))
    r_1 = manner_fn("%s[%s]" %(buf, pos), "%s[%s+1]" %(buf, pos))

    return """
    {
        // This function trashes buf[1..N], leaving the reduction result in buf[0].

        if (%(pos)s < warpSize)
        {
            for (int i = %(pos)s + warpSize; i < %(N)s; i += warpSize)
            {
                %(buf)s[%(pos)s] = %(loop_line)s;
            }
            if (%(pos)s < 16)
            {
                //reduce so that %(pos)s 0 has the sum of everything
                if(%(pos)s + 16 < %(N)s)
                    %(buf)s[%(pos)s] = %(r_16)s;
                if(%(pos)s + 8 < %(N)s)
                    %(buf)s[%(pos)s] = %(r_8)s;
                if(%(pos)s + 4 < %(N)s)
                    %(buf)s[%(pos)s] = %(r_4)s;
                if(%(pos)s + 2 < %(N)s)
                    %(buf)s[%(pos)s] = %(r_2)s;
                if(%(pos)s + 1 < %(N)s)
                    %(buf)s[%(pos)s] = %(r_1)s;
            }
        }
    }
    """ % locals()

@code_version(inline_reduce.code_version)
def inline_reduce_max(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count, lambda a, b: "max(%s, %s)"%(a,b))

@code_version(inline_reduce.code_version)
def inline_reduce_sum(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count, lambda a, b: "%s + %s"%(a,b))

@code_version(inline_reduce.code_version)
def inline_reduce_min(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count, lambda a, b: "min(%s, %s)"%(a,b))

@code_version(inline_reduce.code_version)
def inline_reduce_prod(N, buf, pos, count):
    return inline_reduce(N, buf, pos, count, lambda a, b: "%s * %s"%(a,b))


@code_version((2,) + inline_reduce_max.code_version + inline_reduce_sum.code_version)
def inline_softmax(N, buf, buf2, threadPos, threadCount):
    """

    :param N: length of the buffer
    :param threadPos: index of executing thread
    :param threadCount: number of executing threads

    :Precondition: buf and buf2 contain two identical copies of the input to softmax
    :Postcondition: buf contains the softmax, buf2 contains un-normalized softmax

    :note: buf and buf2 should be in gpu shared memory, we access it many times.

    :note2: We use __i as an int variable in a loop
    """
    return [
            #get max of buf (trashing all but buf[0])
            inline_reduce_max(N, buf, threadPos, threadCount),
            '__syncthreads()',
            'float row_max = '+buf+'[0]',
            '__syncthreads()',
            'for(int __i='+threadPos+'; __i<'+N+'; __i+='+threadCount+'){',
                buf+'[__i] = exp('+buf2+'[__i] - row_max)',
                buf2+'[__i] = '+buf+'[__i]',
            '}',
            '__syncthreads()',
            inline_reduce_sum(N, buf, threadPos, threadCount),
            '__syncthreads()',
            'float row_sum = '+buf+'[0]',
            '__syncthreads()',
            # divide each exp() result by the sum to complete the job.
            'for(int __i='+threadPos+'; __i<'+N+'; __i+='+threadCount+'){',
                buf+'[__i] = '+buf2+'[__i] / row_sum',
            '}',
            '__syncthreads()',
            ]
