"""
Tests of printing functionality
"""


import logging
import StringIO

from nose.plugins.skip import SkipTest

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

    s = StringIO.StringIO()
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

    s = StringIO.StringIO()
    new_handler = logging.StreamHandler(s)
    new_handler.setLevel(logging.DEBUG)
    orig_handler = theano.logging_default_handler

    theano.printing.pydotprint(f, max_label_size=5,
                               print_output_file=False,
                               assert_nb_all_strings=6)


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
        print '--' + mis + '--'
        print '--' + reference + '--'

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
    s = StringIO.StringIO()
    debugprint(G, file=s, ids='int')
    s = s.getvalue()
    # The additional white space are needed!
    reference = """Elemwise{add,no_inplace} [@0] ''   
 |Elemwise{add,no_inplace} [@1] 'C'   
 | |A [@2]
 | |B [@3]
 |Elemwise{add,no_inplace} [@4] ''   
   |D [@5]
   |E [@6]
"""

    if s != reference:
        print '--' + s + '--'
        print '--' + reference + '--'

    assert s == reference

    # test ids=CHAR
    s = StringIO.StringIO()
    debugprint(G, file=s, ids='CHAR')
    s = s.getvalue()
    # The additional white space are needed!
    reference = """Elemwise{add,no_inplace} [@A] ''   
 |Elemwise{add,no_inplace} [@B] 'C'   
 | |A [@C]
 | |B [@D]
 |Elemwise{add,no_inplace} [@E] ''   
   |D [@F]
   |E [@G]
"""

    if s != reference:
        print '--' + s + '--'
        print '--' + reference + '--'

    assert s == reference

    # test ids=CHAR, stop_on_name=True
    s = StringIO.StringIO()
    debugprint(G, file=s, ids='CHAR', stop_on_name=True)
    s = s.getvalue()
    # The additional white space are needed!
    reference = """Elemwise{add,no_inplace} [@A] ''   
 |Elemwise{add,no_inplace} [@B] 'C'   
 |Elemwise{add,no_inplace} [@C] ''   
   |D [@D]
   |E [@E]
"""

    if s != reference:
        print '--' + s + '--'
        print '--' + reference + '--'

    assert s == reference

    # test ids=
    s = StringIO.StringIO()
    debugprint(G, file=s, ids='')
    s = s.getvalue()
    # The additional white space are needed!
    reference = """Elemwise{add,no_inplace}  ''   
 |Elemwise{add,no_inplace}  'C'   
 | |A 
 | |B 
 |Elemwise{add,no_inplace}  ''   
   |D 
   |E 
"""
    if s != reference:
        print '--' + s + '--'
        print '--' + reference + '--'

    assert s == reference
