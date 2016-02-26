import theano
import theano.tensor as T

def test_debugprint():
    k = T.iscalar("k")
    A = T.vector("A")

    # Symbolic description of the result
    result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                                  outputs_info=T.ones_like(A),
                                  non_sequences=A,
                                  n_steps=k,
                                  name="scan")

    final_result = result[-1]

    # compiled function that returns A**k
    power = theano.function(inputs=[A, k],
                            outputs=final_result,
                            updates=updates,
                            mode='DebugMode')

    #a = theano.printing.debugprint(power, file="str")
    #a = theano.compile.debugmode.debugprint(power,
    #                                        prefix="test")

    #print(a)
    theano.printing.debugprint(power)
    print power(range(10), 2)
    print power(range(10), 4)

test_debugprint()
