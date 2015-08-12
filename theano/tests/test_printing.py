"""
Tests of printing functionality
"""
from __future__ import print_function
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
    assert isinstance(ret, str)


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

    # just test that it work
    debugprint(G)

    # test ids=int
    s = StringIO()
    debugprint(G, file=s, ids='int')
    s = s.getvalue()
    # The additional white space are needed!
    reference = '\n'.join([
        "Elemwise{add,no_inplace} [@0] ''   ",
        " |Elemwise{add,no_inplace} [@1] 'C'   ",
        " | |A [@2]",
        " | |B [@3]",
        " |Elemwise{add,no_inplace} [@4] ''   ",
        "   |D [@5]",
        "   |E [@6]",
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
        "Elemwise{add,no_inplace} [@A] ''   ",
        " |Elemwise{add,no_inplace} [@B] 'C'   ",
        " | |A [@C]",
        " | |B [@D]",
        " |Elemwise{add,no_inplace} [@E] ''   ",
        "   |D [@F]",
        "   |E [@G]",
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
        "Elemwise{add,no_inplace} [@A] ''   ",
        " |Elemwise{add,no_inplace} [@B] 'C'   ",
        " |Elemwise{add,no_inplace} [@C] ''   ",
        "   |D [@D]",
        "   |E [@E]",
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

    expected_output = """Subtensor{int64} [@A] ''
     |Subtensor{int64::} [@B] ''
     | |for{cpu,scan_fn} [@C] ''
     | | |k [@D]
     | | |IncSubtensor{Set;:int64:} [@E] ''
     | | | |Alloc [@F] ''
     | | | | |TensorConstant{0.0} [@G]
     | | | | |Elemwise{add,no_inplace} [@H] ''
     | | | | | |k [@D]
     | | | | | |Subtensor{int64} [@I] ''
     | | | | |   |Shape [@J] ''
     | | | | |   | |Rebroadcast{0} [@K] ''
     | | | | |   |   |DimShuffle{x,0} [@L] ''
     | | | | |   |     |Elemwise{second,no_inplace} [@M] ''
     | | | | |   |       |A [@N]
     | | | | |   |       |DimShuffle{x} [@O] ''
     | | | | |   |         |TensorConstant{1.0} [@P]
     | | | | |   |Constant{0} [@Q]
     | | | | |Subtensor{int64} [@R] ''
     | | | |   |Shape [@S] ''
     | | | |   | |Rebroadcast{0} [@K] ''
     | | | |   |Constant{1} [@T]
     | | | |Rebroadcast{0} [@K] ''
     | | | |ScalarFromTensor [@U] ''
     | | |   |Subtensor{int64} [@I] ''
     | | |A [@N]
     | |Constant{1} [@V]
     |Constant{-1} [@W]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [@C] ''
     >Elemwise{mul,no_inplace} [@X] ''
     > |<TensorType(float64, vector)> [@Y] -> [@E]
     > |A_copy [@Z] -> [@N]"""

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

    expected_output = """Sum{acc_dtype=float64} [@A] ''
     |for{cpu,scan_fn} [@B] ''
       |Elemwise{minimum,no_inplace} [@C] ''
       | |Subtensor{int64} [@D] ''
       | | |Shape [@E] ''
       | | | |Subtensor{int64::} [@F] 'coefficients[0:]'
       | | |   |coefficients [@G]
       | | |   |Constant{0} [@H]
       | | |Constant{0} [@I]
       | |Subtensor{int64} [@J] ''
       |   |Shape [@K] ''
       |   | |Subtensor{int64::} [@L] ''
       |   |   |ARange{dtype='int16'} [@M] ''
       |   |   | |TensorConstant{0} [@N]
       |   |   | |TensorConstant{10000} [@O]
       |   |   | |TensorConstant{1} [@P]
       |   |   |Constant{0} [@Q]
       |   |Constant{0} [@R]
       |Subtensor{:int64:} [@S] ''
       | |Subtensor{int64::} [@F] 'coefficients[0:]'
       | |ScalarFromTensor [@T] ''
       |   |Elemwise{minimum,no_inplace} [@C] ''
       |Subtensor{:int64:} [@U] ''
       | |Subtensor{int64::} [@L] ''
       | |ScalarFromTensor [@V] ''
       |   |Elemwise{minimum,no_inplace} [@C] ''
       |Elemwise{minimum,no_inplace} [@C] ''
       |x [@W]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [@B] ''
     >Elemwise{mul,no_inplace} [@X] ''
     > |coefficients[t] [@Y] -> [@S]
     > |Elemwise{pow,no_inplace} [@Z] ''
     >   |x_copy [@BA] -> [@W]
     >   |<TensorType(int16, scalar)> [@BB] -> [@U]"""

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

    expected_output = """Sum{acc_dtype=float64} [@A] ''
     |for{cpu,scan_fn} [@B] ''
       |Elemwise{minimum,no_inplace} [@C] ''
       | |Subtensor{int64} [@D] ''
       | | |Shape [@E] ''
       | | | |Subtensor{int64::} [@F] 'coefficients[0:]'
       | | |   |coefficients [@G]
       | | |   |Constant{0} [@H]
       | | |Constant{0} [@I]
       | |Subtensor{int64} [@J] ''
       |   |Shape [@K] ''
       |   | |Subtensor{int64::} [@L] ''
       |   |   |ARange{dtype='int8'} [@M] ''
       |   |   | |TensorConstant{0} [@N]
       |   |   | |TensorConstant{10} [@O]
       |   |   | |TensorConstant{1} [@P]
       |   |   |Constant{0} [@Q]
       |   |Constant{0} [@R]
       |Subtensor{:int64:} [@S] ''
       | |Subtensor{int64::} [@F] 'coefficients[0:]'
       | |ScalarFromTensor [@T] ''
       |   |Elemwise{minimum,no_inplace} [@C] ''
       |Subtensor{:int64:} [@U] ''
       | |Subtensor{int64::} [@L] ''
       | |ScalarFromTensor [@V] ''
       |   |Elemwise{minimum,no_inplace} [@C] ''
       |Elemwise{minimum,no_inplace} [@C] ''
       |A [@W]
       |k [@X]

    Inner graphs of the scan ops:

    for{cpu,scan_fn} [@B] ''
     >Elemwise{mul,no_inplace} [@Y] ''
     > |DimShuffle{x} [@Z] ''
     > | |coefficients[t] [@BA] -> [@S]
     > |Elemwise{pow,no_inplace} [@BB] ''
     >   |Subtensor{int64} [@BC] ''
     >   | |Subtensor{int64::} [@BD] ''
     >   | | |for{cpu,scan_fn} [@BE] ''
     >   | | | |k_copy [@BF] -> [@X]
     >   | | | |IncSubtensor{Set;:int64:} [@BG] ''
     >   | | | | |Alloc [@BH] ''
     >   | | | | | |TensorConstant{0.0} [@BI]
     >   | | | | | |Elemwise{add,no_inplace} [@BJ] ''
     >   | | | | | | |k_copy [@BF] -> [@X]
     >   | | | | | | |Subtensor{int64} [@BK] ''
     >   | | | | | |   |Shape [@BL] ''
     >   | | | | | |   | |Rebroadcast{0} [@BM] ''
     >   | | | | | |   |   |DimShuffle{x,0} [@BN] ''
     >   | | | | | |   |     |Elemwise{second,no_inplace} [@BO] ''
     >   | | | | | |   |       |A_copy [@BP] -> [@W]
     >   | | | | | |   |       |DimShuffle{x} [@BQ] ''
     >   | | | | | |   |         |TensorConstant{1.0} [@BR]
     >   | | | | | |   |Constant{0} [@BS]
     >   | | | | | |Subtensor{int64} [@BT] ''
     >   | | | | |   |Shape [@BU] ''
     >   | | | | |   | |Rebroadcast{0} [@BM] ''
     >   | | | | |   |Constant{1} [@BV]
     >   | | | | |Rebroadcast{0} [@BM] ''
     >   | | | | |ScalarFromTensor [@BW] ''
     >   | | | |   |Subtensor{int64} [@BK] ''
     >   | | | |A_copy [@BP] -> [@W]
     >   | | |Constant{1} [@BX]
     >   | |Constant{-1} [@BY]
     >   |DimShuffle{x} [@BZ] ''
     >     |<TensorType(int8, scalar)> [@CA] -> [@U]

    for{cpu,scan_fn} [@BE] ''
     >Elemwise{mul,no_inplace} [@CB] ''
     > |<TensorType(float64, vector)> [@CC] -> [@BG]
     > |A_copy [@CD] -> [@BP]"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()


def test_scan_debugprint4():

    def fn(a_m2, a_m1, b_m2, b_m1):
        return a_m1 + a_m2, b_m1 + b_m2

    a0 = theano.shared(numpy.arange(2))
    b0 = theano.shared(numpy.arange(2))

    (a, b), _ = theano.scan(
        fn, outputs_info=[{'initial': a0, 'taps': [-2, -1]},
                          {'initial': b0, 'taps': [-2, -1]}],
        n_steps=5)

    final_result = a + b
    output_str = theano.printing.debugprint(final_result, file='str')
    lines = []
    for line in output_str.split('\n'):
        lines += [line]

    expected_output = """Elemwise{add,no_inplace} [@A] ''
     |Subtensor{int64::} [@B] ''
     | |for{cpu,scan_fn}.0 [@C] ''
     | | |TensorConstant{5} [@D]
     | | |IncSubtensor{Set;:int64:} [@E] ''
     | | | |Alloc [@F] ''
     | | | | |TensorConstant{0} [@G]
     | | | | |Elemwise{add,no_inplace} [@H] ''
     | | | |   |TensorConstant{5} [@D]
     | | | |   |Subtensor{int64} [@I] ''
     | | | |     |Shape [@J] ''
     | | | |     | |Subtensor{:int64:} [@K] ''
     | | | |     |   |<TensorType(int64, vector)> [@L]
     | | | |     |   |Constant{2} [@M]
     | | | |     |Constant{0} [@N]
     | | | |Subtensor{:int64:} [@K] ''
     | | | |ScalarFromTensor [@O] ''
     | | |   |Subtensor{int64} [@I] ''
     | | |IncSubtensor{Set;:int64:} [@P] ''
     | |   |Alloc [@Q] ''
     | |   | |TensorConstant{0} [@G]
     | |   | |Elemwise{add,no_inplace} [@R] ''
     | |   |   |TensorConstant{5} [@D]
     | |   |   |Subtensor{int64} [@S] ''
     | |   |     |Shape [@T] ''
     | |   |     | |Subtensor{:int64:} [@U] ''
     | |   |     |   |<TensorType(int64, vector)> [@V]
     | |   |     |   |Constant{2} [@W]
     | |   |     |Constant{0} [@X]
     | |   |Subtensor{:int64:} [@U] ''
     | |   |ScalarFromTensor [@Y] ''
     | |     |Subtensor{int64} [@S] ''
     | |Constant{2} [@Z]
     |Subtensor{int64::} [@BA] ''
       |for{cpu,scan_fn}.1 [@C] ''
       |Constant{2} [@BB]

    Inner graphs of the scan ops:

    for{cpu,scan_fn}.0 [@C] ''
     >Elemwise{add,no_inplace} [@BC] ''
     > |<TensorType(int64, scalar)> [@BD] -> [@E]
     > |<TensorType(int64, scalar)> [@BE] -> [@E]
     >Elemwise{add,no_inplace} [@BF] ''
     > |<TensorType(int64, scalar)> [@BG] -> [@P]
     > |<TensorType(int64, scalar)> [@BH] -> [@P]

    for{cpu,scan_fn}.1 [@C] ''
     >Elemwise{add,no_inplace} [@BC] ''
     >Elemwise{add,no_inplace} [@BF] ''"""

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

    expected_output = """Subtensor{int64} [@A] ''
     |for{cpu,grad_of_scan_fn}.1 [@B] ''
     | |Elemwise{sub,no_inplace} [@C] ''
     | | |Subtensor{int64} [@D] ''
     | | | |Shape [@E] ''
     | | | | |for{cpu,scan_fn} [@F] ''
     | | | |   |k [@G]
     | | | |   |IncSubtensor{Set;:int64:} [@H] ''
     | | | |   | |Alloc [@I] ''
     | | | |   | | |TensorConstant{0.0} [@J]
     | | | |   | | |Elemwise{add,no_inplace} [@K] ''
     | | | |   | | | |k [@G]
     | | | |   | | | |Subtensor{int64} [@L] ''
     | | | |   | | |   |Shape [@M] ''
     | | | |   | | |   | |Rebroadcast{0} [@N] ''
     | | | |   | | |   |   |DimShuffle{x,0} [@O] ''
     | | | |   | | |   |     |Elemwise{second,no_inplace} [@P] ''
     | | | |   | | |   |       |A [@Q]
     | | | |   | | |   |       |DimShuffle{x} [@R] ''
     | | | |   | | |   |         |TensorConstant{1.0} [@S]
     | | | |   | | |   |Constant{0} [@T]
     | | | |   | | |Subtensor{int64} [@U] ''
     | | | |   | |   |Shape [@V] ''
     | | | |   | |   | |Rebroadcast{0} [@N] ''
     | | | |   | |   |Constant{1} [@W]
     | | | |   | |Rebroadcast{0} [@N] ''
     | | | |   | |ScalarFromTensor [@X] ''
     | | | |   |   |Subtensor{int64} [@L] ''
     | | | |   |A [@Q]
     | | | |Constant{0} [@Y]
     | | |TensorConstant{1} [@Z]
     | |Subtensor{:int64:} [@BA] ''
     | | |Subtensor{::int64} [@BB] ''
     | | | |Subtensor{:int64:} [@BC] ''
     | | | | |for{cpu,scan_fn} [@F] ''
     | | | | |Constant{-1} [@BD]
     | | | |Constant{-1} [@BE]
     | | |ScalarFromTensor [@BF] ''
     | |   |Elemwise{sub,no_inplace} [@C] ''
     | |Subtensor{:int64:} [@BG] ''
     | | |Subtensor{:int64:} [@BH] ''
     | | | |Subtensor{::int64} [@BI] ''
     | | | | |for{cpu,scan_fn} [@F] ''
     | | | | |Constant{-1} [@BJ]
     | | | |Constant{-1} [@BK]
     | | |ScalarFromTensor [@BL] ''
     | |   |Elemwise{sub,no_inplace} [@C] ''
     | |Subtensor{::int64} [@BM] ''
     | | |IncSubtensor{Inc;int64::} [@BN] ''
     | | | |Elemwise{second,no_inplace} [@BO] ''
     | | | | |for{cpu,scan_fn} [@BP] ''
     | | | | | |k [@G]
     | | | | | |IncSubtensor{Set;:int64:} [@H] ''
     | | | | | |A [@Q]
     | | | | |DimShuffle{x,x} [@BQ] ''
     | | | |   |TensorConstant{0.0} [@J]
     | | | |IncSubtensor{Inc;int64} [@BR] ''
     | | | | |Elemwise{second,no_inplace} [@BS] ''
     | | | | | |Subtensor{int64::} [@BT] ''
     | | | | | | |for{cpu,scan_fn} [@BP] ''
     | | | | | | |Constant{1} [@BU]
     | | | | | |DimShuffle{x,x} [@BV] ''
     | | | | |   |TensorConstant{0.0} [@J]
     | | | | |Elemwise{second} [@BW] ''
     | | | | | |Subtensor{int64} [@BX] ''
     | | | | | | |Subtensor{int64::} [@BT] ''
     | | | | | | |Constant{-1} [@BY]
     | | | | | |DimShuffle{x} [@BZ] ''
     | | | | |   |Elemwise{second,no_inplace} [@CA] ''
     | | | | |     |Sum{acc_dtype=float64} [@CB] ''
     | | | | |     | |Subtensor{int64} [@BX] ''
     | | | | |     |TensorConstant{1.0} [@S]
     | | | | |Constant{-1} [@BY]
     | | | |Constant{1} [@BU]
     | | |Constant{-1} [@CC]
     | |Alloc [@CD] ''
     | | |TensorConstant{0.0} [@J]
     | | |Elemwise{add,no_inplace} [@CE] ''
     | | | |Elemwise{sub,no_inplace} [@C] ''
     | | | |TensorConstant{1} [@Z]
     | | |Subtensor{int64} [@CF] ''
     | |   |Shape [@CG] ''
     | |   | |A [@Q]
     | |   |Constant{0} [@CH]
     | |A [@Q]
     |Constant{-1} [@CI]

    Inner graphs of the scan ops:

    for{cpu,grad_of_scan_fn}.1 [@B] ''
     >Elemwise{add,no_inplace} [@CJ] ''
     > |Elemwise{mul} [@CK] ''
     > | |<TensorType(float64, vector)> [@CL] -> [@BM]
     > | |A_copy [@CM] -> [@Q]
     > |<TensorType(float64, vector)> [@CN] -> [@BM]
     >Elemwise{add,no_inplace} [@CO] ''
     > |Elemwise{mul} [@CP] ''
     > | |<TensorType(float64, vector)> [@CL] -> [@BM]
     > | |<TensorType(float64, vector)> [@CQ] -> [@BA]
     > |<TensorType(float64, vector)> [@CR] -> [@CD]

    for{cpu,scan_fn} [@F] ''
     >Elemwise{mul,no_inplace} [@CS] ''
     > |<TensorType(float64, vector)> [@CT] -> [@H]
     > |A_copy [@CU] -> [@Q]

    for{cpu,scan_fn} [@F] ''
     >Elemwise{mul,no_inplace} [@CS] ''

    for{cpu,scan_fn} [@F] ''
     >Elemwise{mul,no_inplace} [@CS] ''

    for{cpu,scan_fn} [@BP] ''
     >Elemwise{mul,no_inplace} [@CS] ''

    for{cpu,scan_fn} [@BP] ''
     >Elemwise{mul,no_inplace} [@CS] ''"""

    for truth, out in zip(expected_output.split("\n"), lines):
        assert truth.strip() == out.strip()
