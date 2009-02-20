#!/usr/bin/env python
import cPickle, numpy, unittest
from theano.compile.module import *
import theano.tensor as T
import sys
import theano
#TODO: add test for module.make(member=init_value)
class T_test_module(unittest.TestCase):

    def test_whats_up_with_submembers(self):
        class Blah(FancyModule):
            def __init__(self, stepsize):
                super(Blah, self).__init__()
                self.stepsize = Member(T.value(stepsize))
                x = T.dscalar()
            
                self.step = Method([x], x - self.stepsize)

        B = Blah(0.0)
        b = B.make(mode='FAST_RUN')
        b.step(1.0)
        assert b.stepsize == 0.0

    def test_members_in_list_tuple_or_dict(self):
        """Test that a Member which is only included via a list, tuple or dictionary is still treated as if it
        were a toplevel attribute and not shared
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
            m1.dlx={"x":[x()]}
            m1.dly={"y":[y()]}
            m1.dtx={"x":(x(),)}
            m1.dty={"y":(y(),)}
            m1.ddx={"x":{"x":x()}}
            m1.ddy={"y":{"y":y()}}

            assert isinstance(m1.x,(gof.Result))
            assert isinstance(m1.y,(gof.Result))
            for i in [m1.lx[0], m1.ly[0], m1.llx[0][0], m1.lly[0][0], m1.ltx[0][0], m1.lty[0][0], m1.ldx[0]['x'], m1.ldy[0]['y'], m1.tx[0], m1.ty[0], m1.tlx[0][0], m1.tly[0][0], m1.ttx[0][0], m1.tty[0][0], m1.tdx[0]['x'], m1.tdy[0]['y'], m1.dx['x'], m1.dy['y'], m1.dlx['x'][0], m1.dly['y'][0], m1.dtx['x'][0], m1.dty['y'][0], m1.ddx['x']['x'], m1.ddy['y']['y']]:
                assert isinstance(i,(gof.Result))
                

            inst=m1.make()

            def get_l():
                return [inst.lx, inst.ly, inst.tx, inst.ty, inst.dx, inst.dy, inst.llx, inst.lly, inst.ltx, inst.lty, inst.ldx, inst.ldy, inst.tlx, inst.tly, inst.ttx, inst.tty, inst.tdx, inst.tdy, inst.dly, inst.dlx, inst.dty, inst.dtx, inst.ddy, inst.ddx] 
            def get_l2():
#                return [inst.lx[0], inst.ly[0], inst.tx[0], inst.ty[0], inst.dx['x'], inst.dy['y'], inst.llx[0][0], inst.lly[0][0], inst.ltx[0][0], inst.lty[0][0], inst.ldx[0]['x'], inst.ldy[0]['y'], inst.tlx[0][0], inst.tly[0][0], inst.ttx[0][0], inst.tty[0][0], inst.tdx, inst.tdy, inst.dly, inst.dlx, inst.dty, inst.dtx, inst.ddy, inst.ddx] 
                return [inst.lx, inst.ly, inst.tx, inst.ty, inst.llx[0], inst.lly[0], inst.ltx[0], inst.lty[0], inst.ldx[0], inst.ldy[0], inst.tlx[0], inst.tly[0], inst.ttx[0], inst.tty[0], inst.tdx[0], inst.tdy[0], inst.dly['y'], inst.dlx['x'], inst.dty['y'], inst.dtx['x']]#, inst.ddy['y'], inst.ddx['x']] 


            #test that we can access the data
            inst.x
            inst.y 
            for i in get_l():
                assert i

            #test that we can set a value to the data the get this value
            inst.x=-1
            inst.y=-2
            inst.ldx[0]['x']=-3
            inst.ldy[0]['y']=-4
            inst.tdx[0]['x']=-5
            inst.tdy[0]['y']=-6
            inst.ddx['x']['x']=-7
            inst.ddy['y']['y']=-8
            for i,j in zip(get_l2(),range(len(get_l2()))):
                i[0]=j
            assert inst.x==-1
            assert inst.y==-2
            assert inst.ldx[0]['x']==-3
            assert inst.ldy[0]['y']==-4
            assert inst.tdx[0]['x']==-5
            assert inst.tdy[0]['y']==-6
            assert inst.ddx['x']['x']==-7
            assert inst.ddy['y']['y']==-8
            for i,j in zip(get_l2(),range(len(get_l2()))):
                assert i[0]==j

        local_test(lambda:T.dscalar(),lambda:Member(T.dscalar()))
        local_test(lambda:T.value(1),lambda:Member(T.value(2)))
        local_test(lambda:T.constant(1),lambda:Member(T.constant(2)))
        
    def test_method_in_list_or_dict(self):
        """Test that a Method which is only included via a list or dictionary is still treated as if it
        were a toplevel attribute
        Fred: why we don't do this of direct fct of results?
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
        """Test that Members can be shared an arbitrary number of times between 
        many submodules and internal data structures."""
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
        Fred: the test create different method event if they are shared. What do we want?
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

        print >> sys.stderr, "MODULE TEST IMPLEMENTED BUT WE DON'T KNOW WHAT WE WANT AS A RESULT"

    def test_shared_method_N(self):
        """Test that Methods can be shared an arbitrary number of times between many submodules and
        internal data structures."""
        
    #put them in subModules, sub-sub-Modules, shared between a list and a dict, shared between
    #a list and a submodule with a dictionary, etc...
        print >> sys.stderr, "WARNING MODULE TEST NOT IMPLEMENTED"

    def test_member_method_inputs(self):
        """Test that module Members can be named as Method inputs, in which case the function will
        *not* use the storage allocated for the Module's version of that Member.
        
        si le module a un membre x et qu''une fct un parametre appele x qui n''est pas le membre cela doit etre bien traiter.
        les poids ne change pas

"""
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

    def test_raise_NotImplemented(self):
        c=Component()
        self.assertRaises(NotImplementedError, c.allocate,"")
        self.assertRaises(NotImplementedError, c.build,"","")
        self.assertRaises(NotImplementedError, c.pretty)
        self.assertRaises(NotImplementedError, c.dup)
        c=Composite()
        self.assertRaises(NotImplementedError, c.resolve,"n")
        self.assertRaises(NotImplementedError, c.components)
        self.assertRaises(NotImplementedError, c.components_map)
        self.assertRaises(NotImplementedError, c.get,"n")
        self.assertRaises(NotImplementedError, c.set,"n",1)

def test_pickle():
    """Test that a module can be pickled"""
    M = Module()
    M.x = Member(T.dmatrix())
    M.y = Member(T.dmatrix())
    a = T.dmatrix()
    M.f = Method([a], a + M.x + M.y)
    M.g = Method([a], a * M.x * M.y)

    m = M.make(x=numpy.zeros((4,5)), y=numpy.ones((2,3)))

    m_dup = cPickle.loads(cPickle.dumps(m))

    assert numpy.all(m.x == m_dup.x) and numpy.all(m.y == m_dup.y)

    m_dup.x[0,0] = 3.142
    assert m_dup.f.input_storage[1].data[0,0] == 3.142
    assert m.x[0,0] == 0.0 #ensure that m is not aliased to m_dup

    #check that the unpickled version has the same argument/property aliasing
    assert m_dup.x is m_dup.f.input_storage[1].data
    assert m_dup.y is m_dup.f.input_storage[2].data
    assert m_dup.x is m_dup.g.input_storage[1].data
    assert m_dup.y is m_dup.g.input_storage[2].data


def test_pickle_aliased_memory():
    M = Module()
    M.x = Member(T.dmatrix())
    M.y = Member(T.dmatrix())
    a = T.dmatrix()
    M.f = Method([a], a + M.x + M.y)
    M.g = Method([a], a * M.x * M.y)

    m = M.make(x=numpy.zeros((4,5)), y=numpy.ones((2,3)))
    m.y = m.x[:]
    m_dup = cPickle.loads(cPickle.dumps(m))

    #m's memory is aliased....
    m.x[0,0] = 3.14
    assert m.y[0,0] == 3.14

    #is m_dup's memory aliased?
    m_dup.x[0,0] = 3.14
    assert m_dup.y[0,0] == 3.14

    #m's memory is aliased differently....
    m.y = m.x[1:2]
    m_dup = cPickle.loads(cPickle.dumps(m))

    #is m_dup's memory aliased the same way?
    m.x[1,0] = 3.142
    assert m.y[0,0] == 3.142
    m_dup.x[1,0] = 3.142
    assert m_dup.y[0,0] == 3.142


def test_tuple_members():

    M = Module()
    M.a = (1,1)
    assert isinstance(M.a, tuple)

    class Temp(Module):
        def __init__(self):
            self.a = (1,1)
    M = Temp()
    assert isinstance(M.a, tuple)


if __name__ == '__main__':
    from theano.tests import main
#    main(__file__[:-3])
    main("test_module")
#    t=T_test_module()
#    t.test_shared_members()
#    tests = unittest.TestLoader().loadTestsFromModule("T_test_module")
#    tests.debug()
