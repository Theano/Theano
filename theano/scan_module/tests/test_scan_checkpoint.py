import numpy
import time

import theano
import theano.tensor as T


def example1(checkpoint=True):

    k = T.iscalar("k")
    A = T.vector("A")

    # Symbolic description of the result
    if checkpoint:
        result, updates = theano.scan_with_checkpoints(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=T.ones_like(A),
            non_sequences=A,
            n_steps=k,
            save_every_N=20)
    else:
        result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                                      outputs_info=T.ones_like(A),
                                      non_sequences=A,
                                      n_steps=k)

    # We only care about A**k, but scan has provided us with A**1 through A**k.
    # Discard the values that we don't care about. Scan is smart enough to
    # notice this and not waste memory saving them.
    result = result[-1]

    # compiled function that returns A**k
    start_compile = time.time()
    power = theano.function(inputs=[A, k], outputs=result, updates=updates)
    time_compile = time.time() - start_compile

    start_exec = time.time()
    out = power(range(10), 100)
    time_exec = time.time() - start_exec

    if checkpoint:
        print("Example 1 with checkpoints")
    else:
        print("Example 1 without checkpoints")
    print("Compile time:", time_compile)
    print("Exec time:", time_exec)
    print("Output:", out)


def example2(checkpoint=True):

    up_to = T.iscalar("up_to")

    # define a named function, rather than using lambda
    def accumulate_by_adding(arange_val, sum_to_date):
        return sum_to_date + arange_val
    seq = T.arange(up_to)

    outputs_info = T.as_tensor_variable(numpy.asarray(0, seq.dtype))

    if checkpoint:
        scan_result, scan_updates = theano.scan_with_checkpoints(
            fn=accumulate_by_adding,
            outputs_info=outputs_info,
            sequences=seq,
            save_every_N=10)
    else:
        scan_result, scan_updates = theano.scan(fn=accumulate_by_adding,
                                                outputs_info=outputs_info,
                                                sequences=seq)

    start_compile = time.time()
    triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)
    time_compile = time.time() - start_compile

    start_exec = time.time()
    out = triangular_sequence(100)[-1]
    time_exec = time.time() - start_exec

    if checkpoint:
        print("Example 2 with checkpoints")
    else:
        print("Example 2 without checkpoints")
    print("Compile time:", time_compile)
    print("Exec time:", time_exec)
    print("Output:", out)


def test_scan_checkpoint():
    example1(False)
    example1(True)
    print("----")
    example2(False)
    example2(True)
    print("----")
