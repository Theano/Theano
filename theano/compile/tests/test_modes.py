"""
Test compilation modes
"""
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
        # Use a new instance of ProfileMode instead of 'PROFILE_MODE' to
        # avoid printing a profile mode summary in nose output
        predef_modes.append(ProfileMode())

        # Linkers to use with regular Mode
        linkers = ['c|py', 'c|py_nogc', 'vm', 'vm_nogc', 'cvm', 'cvm_nogc']
        modes = predef_modes + [Mode(linker, 'fast_run') for linker in linkers]

        for mode in modes:

            x = T.matrix()
            y = T.vector()
            f = theano.function([x, y], x + y, mode=mode)
            # test that it runs something
            f([[1, 2], [3, 4]], [5, 6])
            linker_classes_involved.append(f.maker.mode.linker.__class__)
            print 'MODE:', mode, f.maker.mode.linker, 'stop'
        # regression check:
        # there should be
        # - VM_Linker
        # - OpWiseCLinker (FAST_RUN)
        # - WrapLinker (PROFILE_MODE)
        # - PerformLinker (FAST_COMPILE)
        # - DebugMode's Linker  (DEBUG_MODE)
        assert 5 == len(set(linker_classes_involved))

if __name__ == '__main__':
    unittest.main()
