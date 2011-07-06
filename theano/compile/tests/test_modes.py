"""
Test compilation modes
"""
from nose.plugins.skip import SkipTest

import unittest
import theano
import numpy
import random
import numpy.random
from theano.tests  import unittest_tools as utt

import theano.tensor as T


class T_bunch_of_modes(unittest.TestCase):

    def test1(self):
        # this is a quick test after the LazyLinker branch merge
        # to check that all the current modes can still be used.
        linker_classes_involved = []
        for modename in theano.config.__class__.__dict__['mode'].all:

            x = T.matrix()
            y = T.vector()
            f = theano.function([x,y], x+y, mode=modename)
            # test that it runs something
            f([[1,2],[3,4]], [5, 6])
            linker_classes_involved.append(f.maker.mode.linker.__class__)
            print 'MODE:', modename, f.maker.mode.linker, 'stop'
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
