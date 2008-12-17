#!/usr/bin/env python
import unittest
from theano.compile.module import *
import theano.tensor as T
import sys
import theano

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
        assert b.stepsize == 0.0


    def test_no_shared_members(self):
        """Test that a Result cannot become a Member of two connected Modules
        Fred: What is the purpose of this test? Why we should not be able to do it? I think we should be able to share member.
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
        inst1=m1.make()
        inst2=m2.make()

#        inst1.y=1
#        inst2.y=2
        print inst1
        print inst2
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED1"

    def test_members_in_list_tuple_or_dict(self):
        """Test that a Member which is only included via a list, tuple or dictionary is still treated as if it
        were a toplevel attribute and not shared
        Fred: toplevel attribute? do you mean as a toplevel member? m1.x=x work as m1.lx=[x]?
        """

        def local_test(x,y):
            m1=Module()
            m1.x=x()
            m1.y=y()
            m1.lx=[x()]#cast Result]
            m1.ly=[y()]
            m1.llx=[[x()]]#cast Result]
            m1.lly=[[y()]]
            m1.ltx=[(x(),)]
            m1.lty=[(y(),)]
            m1.ldx=[{"x":x()}]
            m1.ldy=[{"y":y()}]
            m1.tx=(x(),)
            m1.ty=(y(),)
            m1.tlx=[(x(),)]
            m1.tly=[(y(),)]
            m1.ttx=((x(),),)
            m1.tty=((y(),),)
            m1.tdx=({"x":x()},)
            m1.tdy=({"y":y()},)
            m1.dx={"x":x()}
            m1.dy={"y":y()}
            m1.dlx=[{"x":x()}]
            m1.dly=[{"y":y()}]
            m1.dtx=({"x":x()},)
            m1.dty=({"y":y()},)
            m1.ddx={"x":{"x":x()}}
            m1.ddy={"y":{"y":y()}}

            inst=m1.make()

            inst.y 
            inst.x
            assert inst.lx
            assert inst.ly
            assert inst.tx
            assert inst.ty
            assert inst.dx
            assert inst.dy
            assert inst.llx
            assert inst.lly
            assert inst.ltx
            assert inst.lty
            assert inst.ldx
            assert inst.ldy
            assert inst.tlx
            assert inst.tly
            assert inst.ttx
            assert inst.tty
            assert inst.tdx
            assert inst.tdy
            assert inst.dly
            assert inst.dlx
            assert inst.dty
            assert inst.dtx
            assert inst.ddy
            assert inst.ddx

        local_test(lambda:T.dscalar(),lambda:Member(T.dscalar()))
        local_test(lambda:T.value(1),lambda:Member(T.value(2)))
        local_test(lambda:T.constant(1),lambda:Member(T.constant(2)))
        
    def test_method_in_list_or_dict(self):
        """Test that a Method which is only included via a list or dictionary is still treated as if it
        were a toplevel attribute
        Fred: why do we promote a list or tuple of fct of result to a Method?
        Fred: why we don't do this of direct fct of results or dict?
        """
        m1=Module()
        x=T.dscalar()
        m1.x=T.dscalar()
        m1.y=Method(x,x*2)
        m1.z=Method([],m1.x*2)
        m1.ly=[Method(x,x*2)]
        m1.lz=[Method([],m1.x*2)]
        m1.ty=(Method(x,x*2),)
        m1.tz=(Method([],m1.x*2),)
        m1.dy={'y':Method(x,x*2)}
        m1.dz={'z':Method([],m1.x*2)}
        m1.lly=[[Method(x,x*2)]]
        m1.llz=[[Method([],m1.x*2)]]
        m1.lty=[(Method(x,x*2),)]
        m1.ltz=[(Method([],m1.x*2),)]
        m1.ldy=[{'y':Method(x,x*2)}]
        m1.ldz=[{'z':Method([],m1.x*2)}]
        m1.tly=([Method(x,x*2)],)
        m1.tlz=([Method([],m1.x*2)],)
        m1.tty=((Method(x,x*2),),)
        m1.ttz=((Method([],m1.x*2),),)
        m1.tdy=({'y':Method(x,x*2)},)
        m1.tdz=({'z':Method([],m1.x*2)},)
        m1.dly={'y':[Method(x,x*2)]}
        m1.dlz={'z':[Method([],m1.x*2)]}
        m1.dty={'y':(Method(x,x*2),)}
        m1.dtz={'z':(Method([],m1.x*2),)}
        m1.ddy={'y':{'y':Method(x,x*2)}}
        m1.ddz={'z':{'z':Method([],m1.x*2)}}

        inst=m1.make()
        inst.x=1
        assert inst.y(2)==4
        assert inst.z()==2
        assert inst.ly[0](2)==4
        assert inst.lz[0]()==2
        assert inst.ty[0](2)==4
        assert inst.tz[0]()==2
        assert inst.dy['y'](2)==4
        assert inst.dz['z']()==2
        for f in inst.lly[0][0], inst.lty[0][0], inst.ldy[0]['y'], inst.tly[0][0], inst.tty[0][0], inst.tdy[0]['y'], inst.dly['y'][0], inst.dty['y'][0], inst.ddy['y']['y']:
            assert f(2)==4
        for f in inst.llz[0][0], inst.ltz[0][0], inst.ldz[0]['z'], inst.tlz[0][0], inst.ttz[0][0], inst.tdz[0]['z'], inst.dlz['z'][0], inst.dtz['z'][0], inst.ddz['z']['z']:
            assert f()==2

        assert isinstance(inst.z,theano.compile.function_module.Function)
        assert isinstance(inst.y,theano.compile.function_module.Function)
        for f in inst.ly,inst.lz,inst.ty,inst.tz:
            assert isinstance(f[0],theano.compile.function_module.Function)
        for f in inst.lly,inst.llz,inst.lty,inst.ltz,inst.tly,inst.tlz,inst.tty,inst.ttz:
            assert isinstance(f[0][0],theano.compile.function_module.Function)
        for f in inst.dly['y'][0],inst.dty['y'][0], inst.dlz['z'][0],inst.dtz['z'][0], inst.ddy['y']['y'], inst.ddz['z']['z']:
            assert isinstance(f,theano.compile.function_module.Function)
            
    def test_shared_members(self):
        """Test that under a variety of tricky conditions, the shared-ness of Results and Members
        is respected."""

        def populate_module(m,x):
            m.x=x
            m.lx=[x]
            m.llx=[[x],[x]]
            m.ltx=[(x,)]
            m.ldx=[{'x':x}]
            m.tx=(x,)
            m.tlx=([x],)
            m.ttx=((x,),)
            m.tdx=({'x':x},)
            m.dx={'x':x}
            m.dlx={'x':[x]}
            m.dtx={'x':(x,)}
            m.ddx={'x':{'x':x}}

        def get_element(i):
            return [i.x,i.lx[0],i.tx[0],i.dx['x'],i.llx[0][0], i.llx[1][0], i.ltx[0][0], i.ldx[0]['x'], i.tlx[0][0], i.tlx[0][0], i.tdx[0]['x'], i.dlx['x'][0], i.dtx['x'][0], i.ddx['x']['x']]
        m1=Module()
        m2=Module()
        x=T.dscalar()
        populate_module(m1,x)
        populate_module(m2,Member(x))
        #m1.x and m2.x should not be shared as their is no hierarchi link between them.
        inst1=m1.make()
        inst2=m2.make()
        m1.m2=m2
        #m1.x and m2.x should be shared as their is a hierarchi link between them.
        inst3=m1.make()
        inst1.x=1
        inst2.x=2
        inst3.x=3
        for f in get_element(inst1):
            assert f==1
        for f in get_element(inst2):
            assert f==2
        for f in get_element(inst3)+get_element(inst3.m2):
            assert f==3

        inst3.m2.x=4
        for f in get_element(inst3)+get_element(inst3.m2):
            assert f==4

    def test_shared_members_N(self):
        """Test that Members can be shared an arbitrary number of times between many submodules and
        internal data structures."""
        def populate_module(m,x):
            m.x=x
            m.lx=[x]
            m.llx=[[x],[x]]
            m.ltx=[(x,)]
            m.ldx=[{'x':x}]
            m.tx=(x,)
            m.tlx=([x],)
            m.ttx=((x,),)
            m.tdx=({'x':x},)
            m.dx={'x':x}
            m.dlx={'x':[x]}
            m.dtx={'x':(x,)}
            m.ddx={'x':{'x':x}}

        def get_element(i):
            return [i.x,i.lx[0],i.tx[0],i.dx['x'],i.llx[0][0], i.llx[1][0], i.ltx[0][0], i.ldx[0]['x'], i.tlx[0][0], i.tlx[0][0], i.tdx[0]['x'], i.dlx['x'][0], i.dtx['x'][0], i.ddx['x']['x']]
        m1=Module()
        m2=Module()
        m3=Module()
        m4=Module()
        x=T.dscalar()
        populate_module(m1,x)
        populate_module(m2,Member(x))
        populate_module(m4,Member(x))
        #m1.x and m2.x should not be shared as their is no hierarchi link between them.
        inst1=m1.make()
        inst2=m2.make()
        m1.m2=m2
        m2.m3=m3
        m3.m4=m4
        #m1.x and m2.x should be shared as their is a hierarchi link between them.
        inst3=m1.make()
        inst1.x=1
        inst2.x=2
        inst3.x=3
        for f in get_element(inst1):
            assert f==1
        for f in get_element(inst2):
            assert f==2
        for f in get_element(inst3)+get_element(inst3.m2)+get_element(inst3.m2.m3.m4):
            assert f==3

        inst3.m2.x=4
        for f in get_element(inst3)+get_element(inst3.m2)+get_element(inst3.m2.m3.m4):
            assert f==4

    def test_shared_method(self):
        """Test that under a variety of tricky conditions, the shared-ness of Results and Methods
        is respected.
        Fred: the test create different method event if they are shared. Do we want this?
        """

        m1=Module()
        m1.x=T.dscalar()
        x=T.dscalar()
        fy=Method(x,x*2)
        fz=Method([],m1.x*2)
        m1.y=fy
        m1.z=fz
        m1.ly=[fy]
        m1.lz=[fz]
        m1.lly=[[fy]]
        m1.llz=[[fz]]
        m1.ty=(fy,)
        m1.tz=(fz,)
        m1.tty=((fy,),)
        m1.ttz=((fz,),)
        m1.dy={'y':fy}
        m1.dz={'z':fz}

        inst=m1.make()
        inst.x=1
        assert inst.y(2)==4
        assert inst.z()==2
        assert inst.ly[0](2)==4
        assert inst.lz[0]()==2
        assert inst.ty[0](2)==4
        assert inst.tz[0]()==2
        assert inst.dy['y'](2)==4
        assert inst.dz['z']()==2
        assert inst.lly[0][0](2)==4
        assert inst.llz[0][0]()==2
        assert inst.tty[0][0](2)==4
        assert inst.ttz[0][0]()==2
        assert isinstance(inst.z,theano.compile.function_module.Function)
        assert isinstance(inst.lz[0],theano.compile.function_module.Function)
        assert isinstance(inst.llz[0][0],theano.compile.function_module.Function)
        assert isinstance(inst.tz[0],theano.compile.function_module.Function)
        assert isinstance(inst.dz['z'],theano.compile.function_module.Function)
        assert isinstance(inst.ttz[0][0],theano.compile.function_module.Function)
        assert isinstance(inst.y,theano.compile.function_module.Function)
        assert isinstance(inst.ly[0],theano.compile.function_module.Function)
        assert isinstance(inst.lly[0][0],theano.compile.function_module.Function)
        assert isinstance(inst.ty[0],theano.compile.function_module.Function)
        assert isinstance(inst.dy['y'],theano.compile.function_module.Function)
        assert isinstance(inst.tty[0][0],theano.compile.function_module.Function)

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

    def test_member_value(self):
        """Test that module Members of Value work correctly. As Result?"""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    def test_member_constant(self):
        """Test that module Members of Constant work correctly.
        As Result with more optimization?"""
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

if __name__ == '__main__':
    from theano.tests import main
#    main(__file__[:-3])
    main("test_module")
#    t=T_test_module()
#    t.test_shared_members()
#    tests = unittest.TestLoader().loadTestsFromModule("T_test_module")
#    tests.debug()
