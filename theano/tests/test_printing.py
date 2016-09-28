"""
Tests of printing functionality
"""
from __future__ import absolute_import, print_function, division
import logging

from nose.plugins.skip import SkipTest
import numpy

from six.moves import StringIO

import theano
import theano.tensor as tensor

from theano.printing import min_informative_str, debugprint


def test_pydotprint_cond_highlight():
    """
    This is a REALLY PARTIAL TEST.

    I did them to help debug stuff.
    """

    # Skip test if pydot is not available.
    if not theano.printing.pydot_imported:
        raise SkipTest('pydot not available')

    x = tensor.dvector()
    f = theano.function([x], x * 2)
    f([1, 2, 3, 4])

    s = StringIO()
    new_handler = logging.StreamHandler(s)
    new_handler.setLevel(logging.DEBUG)
    orig_handler = theano.logging_default_handler

    theano.theano_logger.removeHandler(orig_handler)
    theano.theano_logger.addHandler(new_handler)
    try:
        theano.printing.pydotprint(f, cond_highlight=True,
                                   print_output_file=False)
    finally:
        theano.theano_logger.addHandler(orig_handler)
        theano.theano_logger.removeHandler(new_handler)

    assert (s.getvalue() == 'pydotprint: cond_highlight is set but there'
            ' is no IfElse node in the graph\n')


def test_pydotprint_return_image():
    # Skip test if pydot is not available.
    if not theano.printing.pydot_imported:
        raise SkipTest('pydot not available')

    x = tensor.dvector()
    ret = theano.printing.pydotprint(x * 2, return_image=True)
    assert isinstance(ret, (str, bytes))


def test_pydotprint_variables():
    """
    This is a REALLY PARTIAL TEST.

    I did them to help debug stuff.

    It make sure the code run.
    """

    # Skip test if pydot is not available.
    if not theano.printing.pydot_imported:
        raise SkipTest('pydot not available')

    x = tensor.dvector()

    s = StringIO()
    new_handler = logging.StreamHandler(s)
    new_handler.setLevel(logging.DEBUG)
    orig_handler = theano.logging_default_handler

    theano.theano_logger.removeHandler(orig_handler)
    theano.theano_logger.addHandler(new_handler)
    try:
        theano.printing.pydotprint(x * 2)
        if not theano.printing.pd.__name__ == "pydot_ng":
            theano.printing.pydotprint_variables(x * 2)
    finally:
        theano.theano_logger.addHandler(orig_handler)
        theano.theano_logger.removeHandler(new_handler)


def test_pydotprint_long_name():
    """This is a REALLY PARTIAL TEST.

    It prints a graph where there are variable and apply nodes whose long
    names are different, but not the shortened names.
    We should not merge those nodes in the dot graph.

    """

    # Skip test if pydot is not available.
    if not theano.printing.pydot_imported:
        raise SkipTest('pydot not available')

    x = tensor.dvector()
    mode = theano.compile.mode.get_default_mode().excluding("fusion")
    f = theano.function([x], [x * 2, x + x], mode=mode)
    f([1, 2, 3, 4])

    theano.printing.pydotprint(f, max_label_size=5,
                               print_output_file=False)
    theano.printing.pydotprint([x * 2, x + x],
                               max_label_size=5,
                               print_output_file=False)


def test_pydotprint_profile():
    """Just check that pydotprint does not crash with ProfileMode."""

    # Skip test if pydot is not available.
    if not theano.printing.pydot_imported:
        raise SkipTest('pydot not available')

    A = tensor.matrix()
    f = theano.function([A], A + 1, mode='ProfileMode')
    theano.printing.pydotprint(f, print_output_file=False)


def test_min_informative_str():
    """ evaluates a reference output to make sure the
        min_informative_str function works as intended """

    A = tensor.matrix(name='A')
    B = tensor.matrix(name='B')
    C = A + B
    C.name = 'C'
    D = tensor.matrix(name='D')
    E = tensor.matrix(name='E')

    F = D + E
    G = C + F

    mis = min_informative_str(G).replace("\t", "        ")

    reference = """A. Elemwise{add,no_inplace}
 B. C
 C. Elemwise{add,no_inplace}
  D. D
  E. E"""

    if mis != reference:
        print('--' + mis + '--')
        print('--' + reference + '--')

    assert mis == reference


def test_debugprint():
    A = tensor.matrix(name='A')
    B = tensor.matrix(name='B')
    C = A + B
    C.name = 'C'
    D = tensor.matrix(name='D')
    E = tensor.matrix(name='E')

    F = D + E
    G = C + F
    mode = theano.compile.get_default_mode().including('fusion')
    g = theano.function([A, B, D, E], G, mode=mode)

    # just test that it work
    s = StringIO()
    debugprint(G, file=s)

    # test ids=int
    s = StringIO()
    debugprint(G, file=s, ids='int')
    s = s.getvalue()
    # The additional white space are needed!
    reference = '\n'.join([
        "Elemwise{add,no_inplace} [id 0] ''   ",
        " |Elemwise{add,no_inplace} [id 1] 'C'   ",
        " | |A [id 2]",
        " | |B [id 3]",
        " |Elemwise{add,no_inplace} [id 4] ''   ",
        "   |D [id 5]",
        "   |E [id 6]",
    ]) + '\n'

    if s != reference:
        print('--' + s + '--')
        print('--' + reference + '--')

    assert s == reference

    # test ids=CHAR
    s = StringIO()
    debugprint(G, file=s, ids='CHAR')
    s = s.getvalue()
    # The additional white space are needed!
    reference = "\n".join([
        "Elemwise{add,no_inplace} [id A] ''   ",
        " |Elemwise{add,no_inplace} [id B] 'C'   ",
        " | |A [id C]",
        " | |B [id D]",
        " |Elemwise{add,no_inplace} [id E] ''   ",
        "   |D [id F]",
        "   |E [id G]",
    ]) + '\n'

    if s != reference:
        print('--' + s + '--')
        print('--' + reference + '--')

    assert s == reference

    # test ids=CHAR, stop_on_name=True
    s = StringIO()
    debugprint(G, file=s, ids='CHAR', stop_on_name=True)
    s = s.getvalue()
    # The additional white space are needed!
    reference = '\n'.join([
        "Elemwise{add,no_inplace} [id A] ''   ",
        " |Elemwise{add,no_inplace} [id B] 'C'   ",
        " |Elemwise{add,no_inplace} [id C] ''   ",
        "   |D [id D]",
        "   |E [id E]",
    ]) + '\n'

    if s != reference:
        print('--' + s + '--')
        print('--' + reference + '--')

    assert s == reference

    # test ids=
    s = StringIO()
    debugprint(G, file=s, ids='')
    s = s.getvalue()
    # The additional white space are needed!
    reference = '\n'.join([
        "Elemwise{add,no_inplace}  ''   ",
        " |Elemwise{add,no_inplace}  'C'   ",
        " | |A ",
        " | |B ",
        " |Elemwise{add,no_inplace}  ''   ",
        "   |D ",
        "   |E ",
    ]) + '\n'
    if s != reference:
        print('--' + s + '--')
        print('--' + reference + '--')

    assert s == reference

    # test print_storage=True
    s = StringIO()
    debugprint(g, file=s, ids='', print_storage=True)
    s = s.getvalue()
    # The additional white space are needed!
    reference = '\n'.join([
        "Elemwise{add,no_inplace}  ''   0 [None]",
        " |A  [None]",
        " |B  [None]",
        " |D  [None]",
        " |E  [None]",
    ]) + '\n'
    if s != reference:
        print('--' + s + '--')
        print('--' + reference + '--')

    assert s == reference

    # test clients
    s = StringIO()
    # We must force the mode as otherwise it can change the clients order
    f = theano.function([A, B, D], [A + B, A + B - D],
                        mode='FAST_COMPILE')
    debugprint(f, file=s, print_clients=True)
    s = s.getvalue()
    # The additional white space are needed!
    reference = '\n'.join([
        "Elemwise{add,no_inplace} [id A] ''   0 clients:[('[id B]', 1), ('output', '')]",
        " |A [id D]",
        " |B [id E]",
        "Elemwise{sub,no_inplace} [id B] ''   1",
        " |Elemwise{add,no_inplace} [id A] ''   0 clients:[('[id B]', 1), ('output', '')]",
        " |D [id F]",
    ]) + '\n'
    if s != reference:
        print('--' + s + '--')
        print('--' + reference + '--')

    assert s == reference


def test_scan_debugprint1():
    k = tensor.iscalar("k")
    A = tensor.dvector("A")

    # Symbolic description of the result
    result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                                  outputs_info=tensor.ones_like(A),
                                  non_sequences=A,
                                  n_steps=k)

    final_result = result[-1]
    output_str = theano.printing.debugprint(final_result, file='str')
    lines = []
    for line in output_str.split('\n'):
        lines += [line]

    expected_output = """Subtensor{int64} [id A] ''
     |Subtensor{int64::} [id B] ''
     | |for{cpu,scan_fn} [id C] ''
     | | |k [id D]
     | | |IncSubtensor{Set;:int64:} [id E] ''
     | | | |AllocEmpty{dtype='float64'} [id F] ''
     | | | | |Elemwise{add,no_inplace} [id G] ''
     | | | | | |k [id D]
     | | | | | |Subtensor{int64} [id H] ''
     | | | | |   |Shape [id I] ''
     | | | | |   | |Rebroadcast{0} [id J] ''
     | | | | |   |   |DimShuffle{x,0} [id K] ''
     | | | | |   |     |Elemwise{second,no_inplace} [id L] ''
     | | | | |   |       |A [id M]
     | | | | |   |       |DimShuffle{x} [id N] ''
     | | | | |   |         |TensorConstant{1.0} [id O]
     | | | | |   |Constant{0} [id P]
     | | | | |Subtensor{int64} [id Q] ''
     | | | |   |Shape [id R] ''
     | | | |   | |Rebroadcast{0} [id J] ''
     | | | |   |Constant{1} [id S]
     | | | |Rebroadcast{0} [id J] ''
     | | | |ScalarFromTensor [id T] ''
     | | |   |Subtensor{int64} [id H] ''
     | | |A [id M]
     | |Constant{1} [id S]
     |Constant{-1} [id U]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [id C] ''
     >Elemwise{mul,no_inplace} [id V] ''
     > |<TensorType(float64, vector)> [id W] -> [id E]
     > |A_copy [id X] -> [id M]"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_scan_debugprint2():
    coefficients = theano.tensor.vector("coefficients")
    x = tensor.scalar("x")

    max_coefficients_supported = 10000

    # Generate the components of the polynomial
    components, updates = theano.scan(fn=lambda coefficient, power,
                                      free_variable:
                                      coefficient * (free_variable ** power),
                                      outputs_info=None,
                                      sequences=[
                                          coefficients,
                                          theano.tensor.arange(
                                              max_coefficients_supported)],
                                      non_sequences=x)
    # Sum them up
    polynomial = components.sum()

    output_str = theano.printing.debugprint(polynomial, file='str')
    lines = []
    for line in output_str.split('\n'):
        lines += [line]

    expected_output = """Sum{acc_dtype=float64} [id A] ''
     |for{cpu,scan_fn} [id B] ''
       |Elemwise{minimum,no_inplace} [id C] ''
       | |Subtensor{int64} [id D] ''
       | | |Shape [id E] ''
       | | | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | | |   |coefficients [id G]
       | | |   |Constant{0} [id H]
       | | |Constant{0} [id H]
       | |Subtensor{int64} [id I] ''
       |   |Shape [id J] ''
       |   | |Subtensor{int64::} [id K] ''
       |   |   |ARange{dtype='int64'} [id L] ''
       |   |   | |TensorConstant{0} [id M]
       |   |   | |TensorConstant{10000} [id N]
       |   |   | |TensorConstant{1} [id O]
       |   |   |Constant{0} [id H]
       |   |Constant{0} [id H]
       |Subtensor{:int64:} [id P] ''
       | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | |ScalarFromTensor [id Q] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Subtensor{:int64:} [id R] ''
       | |Subtensor{int64::} [id K] ''
       | |ScalarFromTensor [id S] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Elemwise{minimum,no_inplace} [id C] ''
       |x [id T]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [id B] ''
     >Elemwise{mul,no_inplace} [id U] ''
     > |coefficients[t] [id V] -> [id P]
     > |Elemwise{pow,no_inplace} [id W] ''
     >   |x_copy [id X] -> [id T]
     >   |<TensorType(int64, scalar)> [id Y] -> [id R]"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip(), (truth, out)


def test_scan_debugprint3():
    coefficients = theano.tensor.dvector("coefficients")
    max_coefficients_supported = 10

    k = tensor.iscalar("k")
    A = tensor.dvector("A")

    # compute A**k
    def compute_A_k(A, k):
        # Symbolic description of the result
        result, updates = theano.scan(fn=lambda prior_result,
                                      A: prior_result * A,
                                      outputs_info=tensor.ones_like(A),
                                      non_sequences=A,
                                      n_steps=k)

        A_k = result[-1]

        return A_k

    # Generate the components of the polynomial
    components, updates = theano.scan(fn=lambda coefficient,
                                      power, some_A, some_k:
                                      coefficient *
                                      (compute_A_k(some_A, some_k) ** power),
                                      outputs_info=None,
                                      sequences=[
                                          coefficients,
                                          theano.tensor.arange(
                                              max_coefficients_supported)],
                                      non_sequences=[A, k])
    # Sum them up
    polynomial = components.sum()

    final_result = polynomial

    output_str = theano.printing.debugprint(final_result, file='str')
    lines = []
    for line in output_str.split('\n'):
        lines += [line]

    expected_output = """Sum{acc_dtype=float64} [id A] ''
     |for{cpu,scan_fn} [id B] ''
       |Elemwise{minimum,no_inplace} [id C] ''
       | |Subtensor{int64} [id D] ''
       | | |Shape [id E] ''
       | | | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | | |   |coefficients [id G]
       | | |   |Constant{0} [id H]
       | | |Constant{0} [id H]
       | |Subtensor{int64} [id I] ''
       |   |Shape [id J] ''
       |   | |Subtensor{int64::} [id K] ''
       |   |   |ARange{dtype='int64'} [id L] ''
       |   |   | |TensorConstant{0} [id M]
       |   |   | |TensorConstant{10} [id N]
       |   |   | |TensorConstant{1} [id O]
       |   |   |Constant{0} [id H]
       |   |Constant{0} [id H]
       |Subtensor{:int64:} [id P] ''
       | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | |ScalarFromTensor [id Q] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Subtensor{:int64:} [id R] ''
       | |Subtensor{int64::} [id K] ''
       | |ScalarFromTensor [id S] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Elemwise{minimum,no_inplace} [id C] ''
       |A [id T]
       |k [id U]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [id B] ''
     >Elemwise{mul,no_inplace} [id V] ''
     > |DimShuffle{x} [id W] ''
     > | |coefficients[t] [id X] -> [id P]
     > |Elemwise{pow,no_inplace} [id Y] ''
     >   |Subtensor{int64} [id Z] ''
     >   | |Subtensor{int64::} [id BA] ''
     >   | | |for{cpu,scan_fn} [id BB] ''
     >   | | | |k_copy [id BC] -> [id U]
     >   | | | |IncSubtensor{Set;:int64:} [id BD] ''
     >   | | | | |AllocEmpty{dtype='float64'} [id BE] ''
     >   | | | | | |Elemwise{add,no_inplace} [id BF] ''
     >   | | | | | | |k_copy [id BC] -> [id U]
     >   | | | | | | |Subtensor{int64} [id BG] ''
     >   | | | | | |   |Shape [id BH] ''
     >   | | | | | |   | |Rebroadcast{0} [id BI] ''
     >   | | | | | |   |   |DimShuffle{x,0} [id BJ] ''
     >   | | | | | |   |     |Elemwise{second,no_inplace} [id BK] ''
     >   | | | | | |   |       |A_copy [id BL] -> [id T]
     >   | | | | | |   |       |DimShuffle{x} [id BM] ''
     >   | | | | | |   |         |TensorConstant{1.0} [id BN]
     >   | | | | | |   |Constant{0} [id H]
     >   | | | | | |Subtensor{int64} [id BO] ''
     >   | | | | |   |Shape [id BP] ''
     >   | | | | |   | |Rebroadcast{0} [id BI] ''
     >   | | | | |   |Constant{1} [id BQ]
     >   | | | | |Rebroadcast{0} [id BI] ''
     >   | | | | |ScalarFromTensor [id BR] ''
     >   | | | |   |Subtensor{int64} [id BG] ''
     >   | | | |A_copy [id BL] -> [id T]
     >   | | |Constant{1} [id BQ]
     >   | |Constant{-1} [id BS]
     >   |DimShuffle{x} [id BT] ''
     >     |<TensorType(int64, scalar)> [id BU] -> [id R]

    for{cpu,scan_fn} [id BB] ''
     >Elemwise{mul,no_inplace} [id BV] ''
     > |<TensorType(float64, vector)> [id BW] -> [id BD]
     > |A_copy [id BX] -> [id BL]"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip(), (truth, out)


def test_scan_debugprint4():

    def fn(a_m2, a_m1, b_m2, b_m1):
        return a_m1 + a_m2, b_m1 + b_m2

    a0 = theano.shared(numpy.arange(2, dtype='int64'))
    b0 = theano.shared(numpy.arange(2, dtype='int64'))

    (a, b), _ = theano.scan(
        fn, outputs_info=[{'initial': a0, 'taps': [-2, -1]},
                          {'initial': b0, 'taps': [-2, -1]}],
        n_steps=5)

    final_result = a + b
    output_str = theano.printing.debugprint(final_result, file='str')
    lines = []
    for line in output_str.split('\n'):
        lines += [line]

    expected_output = """Elemwise{add,no_inplace} [id A] ''
     |Subtensor{int64::} [id B] ''
     | |for{cpu,scan_fn}.0 [id C] ''
     | | |TensorConstant{5} [id D]
     | | |IncSubtensor{Set;:int64:} [id E] ''
     | | | |AllocEmpty{dtype='int64'} [id F] ''
     | | | | |Elemwise{add,no_inplace} [id G] ''
     | | | |   |TensorConstant{5} [id D]
     | | | |   |Subtensor{int64} [id H] ''
     | | | |     |Shape [id I] ''
     | | | |     | |Subtensor{:int64:} [id J] ''
     | | | |     |   |<TensorType(int64, vector)> [id K]
     | | | |     |   |Constant{2} [id L]
     | | | |     |Constant{0} [id M]
     | | | |Subtensor{:int64:} [id J] ''
     | | | |ScalarFromTensor [id N] ''
     | | |   |Subtensor{int64} [id H] ''
     | | |IncSubtensor{Set;:int64:} [id O] ''
     | |   |AllocEmpty{dtype='int64'} [id P] ''
     | |   | |Elemwise{add,no_inplace} [id Q] ''
     | |   |   |TensorConstant{5} [id D]
     | |   |   |Subtensor{int64} [id R] ''
     | |   |     |Shape [id S] ''
     | |   |     | |Subtensor{:int64:} [id T] ''
     | |   |     |   |<TensorType(int64, vector)> [id U]
     | |   |     |   |Constant{2} [id L]
     | |   |     |Constant{0} [id M]
     | |   |Subtensor{:int64:} [id T] ''
     | |   |ScalarFromTensor [id V] ''
     | |     |Subtensor{int64} [id R] ''
     | |Constant{2} [id L]
     |Subtensor{int64::} [id W] ''
       |for{cpu,scan_fn}.1 [id C] ''
       |Constant{2} [id L]

    Inner graphs of the scan ops:

    for{cpu,scan_fn}.0 [id C] ''
     >Elemwise{add,no_inplace} [id X] ''
     > |<TensorType(int64, scalar)> [id Y] -> [id E]
     > |<TensorType(int64, scalar)> [id Z] -> [id E]
     >Elemwise{add,no_inplace} [id BA] ''
     > |<TensorType(int64, scalar)> [id BB] -> [id O]
     > |<TensorType(int64, scalar)> [id BC] -> [id O]

    for{cpu,scan_fn}.1 [id C] ''
     >Elemwise{add,no_inplace} [id X] ''
     >Elemwise{add,no_inplace} [id BA] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip(), (truth, out)


def test_scan_debugprint5():

    k = tensor.iscalar("k")
    A = tensor.dvector("A")

    # Symbolic description of the result
    result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                                  outputs_info=tensor.ones_like(A),
                                  non_sequences=A,
                                  n_steps=k)

    final_result = tensor.grad(result[-1].sum(), A)

    output_str = theano.printing.debugprint(final_result, file='str')
    lines = []
    for line in output_str.split('\n'):
        lines += [line]

    expected_output = """Subtensor{int64} [id A] ''
    |for{cpu,grad_of_scan_fn}.1 [id B] ''
    | |Elemwise{sub,no_inplace} [id C] ''
    | | |Subtensor{int64} [id D] ''
    | | | |Shape [id E] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | |   |k [id G]
    | | | |   |IncSubtensor{Set;:int64:} [id H] ''
    | | | |   | |AllocEmpty{dtype='float64'} [id I] ''
    | | | |   | | |Elemwise{add,no_inplace} [id J] ''
    | | | |   | | | |k [id G]
    | | | |   | | | |Subtensor{int64} [id K] ''
    | | | |   | | |   |Shape [id L] ''
    | | | |   | | |   | |Rebroadcast{0} [id M] ''
    | | | |   | | |   |   |DimShuffle{x,0} [id N] ''
    | | | |   | | |   |     |Elemwise{second,no_inplace} [id O] ''
    | | | |   | | |   |       |A [id P]
    | | | |   | | |   |       |DimShuffle{x} [id Q] ''
    | | | |   | | |   |         |TensorConstant{1.0} [id R]
    | | | |   | | |   |Constant{0} [id S]
    | | | |   | | |Subtensor{int64} [id T] ''
    | | | |   | |   |Shape [id U] ''
    | | | |   | |   | |Rebroadcast{0} [id M] ''
    | | | |   | |   |Constant{1} [id V]
    | | | |   | |Rebroadcast{0} [id M] ''
    | | | |   | |ScalarFromTensor [id W] ''
    | | | |   |   |Subtensor{int64} [id K] ''
    | | | |   |A [id P]
    | | | |Constant{0} [id S]
    | | |TensorConstant{1} [id X]
    | |Subtensor{:int64:} [id Y] ''
    | | |Subtensor{::int64} [id Z] ''
    | | | |Subtensor{:int64:} [id BA] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |Constant{-1} [id BB]
    | | | |Constant{-1} [id BB]
    | | |ScalarFromTensor [id BC] ''
    | |   |Elemwise{sub,no_inplace} [id C] ''
    | |Subtensor{:int64:} [id BD] ''
    | | |Subtensor{:int64:} [id BE] ''
    | | | |Subtensor{::int64} [id BF] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |Constant{-1} [id BB]
    | | | |Constant{-1} [id BB]
    | | |ScalarFromTensor [id BG] ''
    | |   |Elemwise{sub,no_inplace} [id C] ''
    | |Subtensor{::int64} [id BH] ''
    | | |IncSubtensor{Inc;int64::} [id BI] ''
    | | | |Elemwise{second,no_inplace} [id BJ] ''
    | | | | |for{cpu,scan_fn} [id BK] ''
    | | | | | |k [id G]
    | | | | | |IncSubtensor{Set;:int64:} [id H] ''
    | | | | | |A [id P]
    | | | | |DimShuffle{x,x} [id BL] ''
    | | | |   |TensorConstant{0.0} [id BM]
    | | | |IncSubtensor{Inc;int64} [id BN] ''
    | | | | |Elemwise{second,no_inplace} [id BO] ''
    | | | | | |Subtensor{int64::} [id BP] ''
    | | | | | | |for{cpu,scan_fn} [id BK] ''
    | | | | | | |Constant{1} [id V]
    | | | | | |DimShuffle{x,x} [id BQ] ''
    | | | | |   |TensorConstant{0.0} [id BM]
    | | | | |Elemwise{second} [id BR] ''
    | | | | | |Subtensor{int64} [id BS] ''
    | | | | | | |Subtensor{int64::} [id BP] ''
    | | | | | | |Constant{-1} [id BB]
    | | | | | |DimShuffle{x} [id BT] ''
    | | | | |   |Elemwise{second,no_inplace} [id BU] ''
    | | | | |     |Sum{acc_dtype=float64} [id BV] ''
    | | | | |     | |Subtensor{int64} [id BS] ''
    | | | | |     |TensorConstant{1.0} [id R]
    | | | | |Constant{-1} [id BB]
    | | | |Constant{1} [id V]
    | | |Constant{-1} [id BB]
    | |Alloc [id BW] ''
    | | |TensorConstant{0.0} [id BM]
    | | |Elemwise{add,no_inplace} [id BX] ''
    | | | |Elemwise{sub,no_inplace} [id C] ''
    | | | |TensorConstant{1} [id X]
    | | |Subtensor{int64} [id BY] ''
    | |   |Shape [id BZ] ''
    | |   | |A [id P]
    | |   |Constant{0} [id S]
    | |A [id P]
    |Constant{-1} [id BB]

    Inner graphs of the scan ops:

    for{cpu,grad_of_scan_fn}.1 [id B] ''
    >Elemwise{add,no_inplace} [id CA] ''
    > |Elemwise{mul} [id CB] ''
    > | |<TensorType(float64, vector)> [id CC] -> [id BH]
    > | |A_copy [id CD] -> [id P]
    > |<TensorType(float64, vector)> [id CE] -> [id BH]
    >Elemwise{add,no_inplace} [id CF] ''
    > |Elemwise{mul} [id CG] ''
    > | |<TensorType(float64, vector)> [id CC] -> [id BH]
    > | |<TensorType(float64, vector)> [id CH] -> [id Y]
    > |<TensorType(float64, vector)> [id CI] -> [id BW]

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CJ] ''
    > |<TensorType(float64, vector)> [id CK] -> [id H]
    > |A_copy [id CL] -> [id P]

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CJ] ''

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CJ] ''

    for{cpu,scan_fn} [id BK] ''
    >Elemwise{mul,no_inplace} [id CJ] ''

    for{cpu,scan_fn} [id BK] ''
    >Elemwise{mul,no_inplace} [id CJ] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip(), (truth, out)


def test_printing_scan():
    # Skip test if pydot is not available.
    if not theano.printing.pydot_imported:
        raise SkipTest('pydot not available')

    def f_pow2(x_tm1):
        return 2 * x_tm1

    state = theano.tensor.scalar('state')
    n_steps = theano.tensor.iscalar('nsteps')
    output, updates = theano.scan(f_pow2,
                                  [],
                                  state,
                                  [],
                                  n_steps=n_steps,
                                  truncate_gradient=-1,
                                  go_backwards=False)
    f = theano.function([state, n_steps],
                        output,
                        updates=updates,
                        allow_input_downcast=True)
    theano.printing.pydotprint(output, scan_graphs=True)
    theano.printing.pydotprint(f, scan_graphs=True)
