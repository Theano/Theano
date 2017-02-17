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
    """Just check that pydotprint does not crash with profile."""

    # Skip test if pydot is not available.
    if not theano.printing.pydot_imported:
        raise SkipTest('pydot not available')

    A = tensor.matrix()
    prof = theano.compile.ProfileStats(atexit_print=False)
    f = theano.function([A], A + 1, profile=prof)
    theano.printing.pydotprint(f, print_output_file=False)
    f([[1]])
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
     | | | | |   |   |InplaceDimShuffle{x,0} [id K] ''
     | | | | |   |     |Elemwise{second,no_inplace} [id L] ''
     | | | | |   |       |A [id M]
     | | | | |   |       |InplaceDimShuffle{x} [id N] ''
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
     | |Constant{1} [id U]
     |Constant{-1} [id V]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [id C] ''
     >Elemwise{mul,no_inplace} [id W] ''
     > |<TensorType(float64, vector)> [id X] -> [id E]
     > |A_copy [id Y] -> [id M]"""

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
       | | |Constant{0} [id I]
       | |Subtensor{int64} [id J] ''
       |   |Shape [id K] ''
       |   | |Subtensor{int64::} [id L] ''
       |   |   |ARange{dtype='int64'} [id M] ''
       |   |   | |TensorConstant{0} [id N]
       |   |   | |TensorConstant{10000} [id O]
       |   |   | |TensorConstant{1} [id P]
       |   |   |Constant{0} [id Q]
       |   |Constant{0} [id R]
       |Subtensor{:int64:} [id S] ''
       | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | |ScalarFromTensor [id T] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Subtensor{:int64:} [id U] ''
       | |Subtensor{int64::} [id L] ''
       | |ScalarFromTensor [id V] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Elemwise{minimum,no_inplace} [id C] ''
       |x [id W]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [id B] ''
     >Elemwise{mul,no_inplace} [id X] ''
     > |coefficients[t] [id Y] -> [id S]
     > |Elemwise{pow,no_inplace} [id Z] ''
     >   |x_copy [id BA] -> [id W]
     >   |<TensorType(int64, scalar)> [id BB] -> [id U]"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


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
       | | |Constant{0} [id I]
       | |Subtensor{int64} [id J] ''
       |   |Shape [id K] ''
       |   | |Subtensor{int64::} [id L] ''
       |   |   |ARange{dtype='int64'} [id M] ''
       |   |   | |TensorConstant{0} [id N]
       |   |   | |TensorConstant{10} [id O]
       |   |   | |TensorConstant{1} [id P]
       |   |   |Constant{0} [id Q]
       |   |Constant{0} [id R]
       |Subtensor{:int64:} [id S] ''
       | |Subtensor{int64::} [id F] 'coefficients[0:]'
       | |ScalarFromTensor [id T] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Subtensor{:int64:} [id U] ''
       | |Subtensor{int64::} [id L] ''
       | |ScalarFromTensor [id V] ''
       |   |Elemwise{minimum,no_inplace} [id C] ''
       |Elemwise{minimum,no_inplace} [id C] ''
       |A [id W]
       |k [id X]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [id B] ''
     >Elemwise{mul,no_inplace} [id Y] ''
     > |InplaceDimShuffle{x} [id Z] ''
     > | |coefficients[t] [id BA] -> [id S]
     > |Elemwise{pow,no_inplace} [id BB] ''
     >   |Subtensor{int64} [id BC] ''
     >   | |Subtensor{int64::} [id BD] ''
     >   | | |for{cpu,scan_fn} [id BE] ''
     >   | | | |k_copy [id BF] -> [id X]
     >   | | | |IncSubtensor{Set;:int64:} [id BG] ''
     >   | | | | |AllocEmpty{dtype='float64'} [id BH] ''
     >   | | | | | |Elemwise{add,no_inplace} [id BI] ''
     >   | | | | | | |k_copy [id BF] -> [id X]
     >   | | | | | | |Subtensor{int64} [id BJ] ''
     >   | | | | | |   |Shape [id BK] ''
     >   | | | | | |   | |Rebroadcast{0} [id BL] ''
     >   | | | | | |   |   |InplaceDimShuffle{x,0} [id BM] ''
     >   | | | | | |   |     |Elemwise{second,no_inplace} [id BN] ''
     >   | | | | | |   |       |A_copy [id BO] -> [id W]
     >   | | | | | |   |       |InplaceDimShuffle{x} [id BP] ''
     >   | | | | | |   |         |TensorConstant{1.0} [id BQ]
     >   | | | | | |   |Constant{0} [id BR]
     >   | | | | | |Subtensor{int64} [id BS] ''
     >   | | | | |   |Shape [id BT] ''
     >   | | | | |   | |Rebroadcast{0} [id BL] ''
     >   | | | | |   |Constant{1} [id BU]
     >   | | | | |Rebroadcast{0} [id BL] ''
     >   | | | | |ScalarFromTensor [id BV] ''
     >   | | | |   |Subtensor{int64} [id BJ] ''
     >   | | | |A_copy [id BO] -> [id W]
     >   | | |Constant{1} [id BW]
     >   | |Constant{-1} [id BX]
     >   |InplaceDimShuffle{x} [id BY] ''
     >     |<TensorType(int64, scalar)> [id BZ] -> [id U]

    for{cpu,scan_fn} [id BE] ''
     >Elemwise{mul,no_inplace} [id CA] ''
     > |<TensorType(float64, vector)> [id CB] -> [id BG]
     > |A_copy [id CC] -> [id BO]"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


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
     | |   |     |   |Constant{2} [id V]
     | |   |     |Constant{0} [id W]
     | |   |Subtensor{:int64:} [id T] ''
     | |   |ScalarFromTensor [id X] ''
     | |     |Subtensor{int64} [id R] ''
     | |Constant{2} [id Y]
     |Subtensor{int64::} [id Z] ''
       |for{cpu,scan_fn}.1 [id C] ''
       |Constant{2} [id BA]

    Inner graphs of the scan ops:

    for{cpu,scan_fn}.0 [id C] ''
     >Elemwise{add,no_inplace} [id BB] ''
     > |<TensorType(int64, scalar)> [id BC] -> [id E]
     > |<TensorType(int64, scalar)> [id BD] -> [id E]
     >Elemwise{add,no_inplace} [id BE] ''
     > |<TensorType(int64, scalar)> [id BF] -> [id O]
     > |<TensorType(int64, scalar)> [id BG] -> [id O]

    for{cpu,scan_fn}.1 [id C] ''
     >Elemwise{add,no_inplace} [id BB] ''
     >Elemwise{add,no_inplace} [id BE] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


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
    | | | |   | | |   |   |InplaceDimShuffle{x,0} [id N] ''
    | | | |   | | |   |     |Elemwise{second,no_inplace} [id O] ''
    | | | |   | | |   |       |A [id P]
    | | | |   | | |   |       |InplaceDimShuffle{x} [id Q] ''
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
    | | | |Constant{0} [id X]
    | | |TensorConstant{1} [id Y]
    | |Subtensor{:int64:} [id Z] ''
    | | |Subtensor{::int64} [id BA] ''
    | | | |Subtensor{:int64:} [id BB] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |Constant{-1} [id BC]
    | | | |Constant{-1} [id BD]
    | | |ScalarFromTensor [id BE] ''
    | |   |Elemwise{sub,no_inplace} [id C] ''
    | |Subtensor{:int64:} [id BF] ''
    | | |Subtensor{:int64:} [id BG] ''
    | | | |Subtensor{::int64} [id BH] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |Constant{-1} [id BI]
    | | | |Constant{-1} [id BJ]
    | | |ScalarFromTensor [id BK] ''
    | |   |Elemwise{sub,no_inplace} [id C] ''
    | |Subtensor{::int64} [id BL] ''
    | | |IncSubtensor{Inc;int64::} [id BM] ''
    | | | |Elemwise{second,no_inplace} [id BN] ''
    | | | | |for{cpu,scan_fn} [id F] ''
    | | | | |InplaceDimShuffle{x,x} [id BO] ''
    | | | |   |TensorConstant{0.0} [id BP]
    | | | |IncSubtensor{Inc;int64} [id BQ] ''
    | | | | |Elemwise{second,no_inplace} [id BR] ''
    | | | | | |Subtensor{int64::} [id BS] ''
    | | | | | | |for{cpu,scan_fn} [id F] ''
    | | | | | | |Constant{1} [id BT]
    | | | | | |InplaceDimShuffle{x,x} [id BU] ''
    | | | | |   |TensorConstant{0.0} [id BP]
    | | | | |Elemwise{second} [id BV] ''
    | | | | | |Subtensor{int64} [id BW] ''
    | | | | | | |Subtensor{int64::} [id BS] ''
    | | | | | | |Constant{-1} [id BX]
    | | | | | |InplaceDimShuffle{x} [id BY] ''
    | | | | |   |Elemwise{second,no_inplace} [id BZ] ''
    | | | | |     |Sum{acc_dtype=float64} [id CA] ''
    | | | | |     | |Subtensor{int64} [id BW] ''
    | | | | |     |TensorConstant{1.0} [id R]
    | | | | |Constant{-1} [id BX]
    | | | |Constant{1} [id BT]
    | | |Constant{-1} [id CB]
    | |Alloc [id CC] ''
    | | |TensorConstant{0.0} [id BP]
    | | |Elemwise{add,no_inplace} [id CD] ''
    | | | |Elemwise{sub,no_inplace} [id C] ''
    | | | |TensorConstant{1} [id Y]
    | | |Subtensor{int64} [id CE] ''
    | |   |Shape [id CF] ''
    | |   | |A [id P]
    | |   |Constant{0} [id CG]
    | |A [id P]
    |Constant{-1} [id CH]

    Inner graphs of the scan ops:

    for{cpu,grad_of_scan_fn}.1 [id B] ''
    >Elemwise{add,no_inplace} [id CI] ''
    > |Elemwise{mul} [id CJ] ''
    > | |<TensorType(float64, vector)> [id CK] -> [id BL]
    > | |A_copy [id CL] -> [id P]
    > |<TensorType(float64, vector)> [id CM] -> [id BL]
    >Elemwise{add,no_inplace} [id CN] ''
    > |Elemwise{mul} [id CO] ''
    > | |<TensorType(float64, vector)> [id CK] -> [id BL]
    > | |<TensorType(float64, vector)> [id CP] -> [id Z]
    > |<TensorType(float64, vector)> [id CQ] -> [id CC]

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CR] ''
    > |<TensorType(float64, vector)> [id CS] -> [id H]
    > |A_copy [id CT] -> [id P]

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CR] ''

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CR] ''

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CR] ''

    for{cpu,scan_fn} [id F] ''
    >Elemwise{mul,no_inplace} [id CR] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


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


def test_subtensor():
    x = theano.tensor.dvector()
    y = x[1]
    assert theano.pp(y) == "<TensorType(float64, vector)>[Constant{1}]"
