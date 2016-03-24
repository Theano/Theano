"""
Test compilation modes
"""
from __future__ import absolute_import, print_function, division
import copy
import unittest

import theano
import theano.tensor as T
from theano.compile import Mode, ProfileMode


class T_bunch_of_modes(unittest.TestCase):

    def test1(self):
        # this is a quick test after the LazyLinker branch merge
        # to check that all the current modes can still be used.
        linker_classes_involved = []

        predef_modes = ['FAST_COMPILE', 'FAST_RUN', 'DEBUG_MODE']
        # Use a new instance of ProfileMode instead of 'ProfileMode' to
        # avoid printing a profile mode summary in nose output
        predef_modes.append(ProfileMode())

        # Linkers to use with regular Mode
        if theano.config.cxx:
            linkers = ['py', 'c|py', 'c|py_nogc', 'vm', 'vm_nogc',
                       'cvm', 'cvm_nogc']
        else:
            linkers = ['py', 'c|py', 'c|py_nogc', 'vm', 'vm_nogc']
        modes = predef_modes + [Mode(linker, 'fast_run') for linker in linkers]

        for mode in modes:
            x = T.matrix()
            y = T.vector()
            f = theano.function([x, y], x + y, mode=mode)
            # test that it runs something
            f([[1, 2], [3, 4]], [5, 6])
            linker_classes_involved.append(f.maker.mode.linker.__class__)
            # print 'MODE:', mode, f.maker.mode.linker, 'stop'

        # regression check:
        # there should be
        # - VM_Linker
        # - OpWiseCLinker (FAST_RUN)
        # - WrapLinker ("ProfileMode")
        # - PerformLinker (FAST_COMPILE)
        # - DebugMode's Linker  (DEBUG_MODE)
        assert 5 == len(set(linker_classes_involved))


class T_ProfileMode_WrapLinker(unittest.TestCase):
    def test_1(self):
        # First, compile a function with a new ProfileMode() object
        # No need to call that function
        x = T.matrix()
        mode = ProfileMode()
        theano.function([x], x * 2, mode=mode)

        # Then, build a mode with the same linker, and a modified optimizer
        default_mode = theano.compile.mode.get_default_mode()
        modified_mode = default_mode.including('specialize')

        # The following line used to fail, with Python 2.4, in July 2012,
        # because an fgraph was associated to the default linker
        copy.deepcopy(modified_mode)

        # More straightforward test
        linker = theano.compile.mode.get_default_mode().linker
        assert not hasattr(linker, "fgraph") or linker.fgraph is None


if __name__ == '__main__':
    unittest.main()
