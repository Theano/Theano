import unittest
from theano.compile.module import *
import theano.tensor as T
import sys

class T_test_module(unittest.TestCase):

    def test_whats_up_with_submembers(self):
        class Blah(FancyModule):
            def __init__(self, stepsize):
                super(Blah, self).__init__(self)
                self.stepsize = Member(T.value(stepsize))
                x = T.dscalar()
            
                self.step = Method([x], x - self.stepsize)

        B = Blah(0.0)
        b = B.make(mode='FAST_RUN')
        b.step(1.0)
        print b.stepsize
        assert b.stepsize == 0.0


    def test_no_shared_members(self):
        """Test that a Result cannot become a Member of two connected Modules
        FRED: What is the purpose of this test? Why we should not be able to do it?
        Right now it seem to work when it should not.
        """
        x=T.dscalar()
        y=Member(T.dscalar())
        m1=Module()
        m2=Module()
        m1.x=x
        m2.x=x
        m1.y=y
        m2.y=y
        m2.m1=m1
        m2.make()
        m1.make()
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED1"

    def test_members_in_list_or_dict(self):
        """Test that a Member which is only included via a list or dictionary is still treated as if it
        were a toplevel attribute
        Fred: toplevel attribute? toplevel member?
        """

        x=T.dscalar()
        y=Member(T.dscalar())
        m1=Module()
        m1.lx=[x]
        m1.ly=[y]
        m1.dx={"x":x}
        m1.dy={"y":y}
        m1.x=x
        m1.y=y
        inst=m1.make()
        print m1
        print inst
        assert inst.lx
        assert inst.ly
        self.assertRaises(AttributeError, inst.__getattr__, x)
        self.assertRaises(AttributeError, inst.__getattr__, y)#FRED why this raise an exception?
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED2"
        
    def test_method_in_list_or_dict(self):
        """Test that a Method which is only included via a list or dictionary is still treated as if it
        were a toplevel attribute"""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    def test_shared_members(self):
        """Test that under a variety of tricky conditions, the shared-ness of Results and Members
        is respected."""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    #put them in subModules, sub-sub-Modules, shared between a list and a dict, shared between
    #a list and a submodule with a dictionary, etc...

    def test_shared_members_N(self):
        """Test that Members can be shared an arbitrary number of times between many submodules and
        internal data structures."""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    #put them in subModules, sub-sub-Modules, shared between a list and a dict, shared between
    #a list and a submodule with a dictionary, etc...

    def test_shared_method(self):
        """Test that under a variety of tricky conditions, the shared-ness of Results and Methods
        is respected."""

        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"
    #put them in subModules, sub-sub-Modules, shared between a list and a dict, shared between
    #a list and a submodule with a dictionary, etc...

    def test_shared_method_N(self):
        """Test that Methods can be shared an arbitrary number of times between many submodules and
        internal data structures."""
        
    #put them in subModules, sub-sub-Modules, shared between a list and a dict, shared between
    #a list and a submodule with a dictionary, etc...
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    def test_member_method_inputs(self):
        """Test that module Members can be named as Method inputs, in which case the function will
        *not* use the storage allocated for the Module's version of that Member."""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    def test_member_input_flags(self):
        """Test that we can manipulate the mutable, strict, etc. flags (see SymbolicInput) of
        Method inputs"""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    def test_member_output_flags(self):
        """Test that we can manipulate the output flags (just 'borrow' I think, see SymbolicOutput)
        of Method outputs"""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    def test_sanity_check_mode(self):
        """Test that Module.make(self) can take the same list of Modes that function can, so we can
        debug modules"""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"
