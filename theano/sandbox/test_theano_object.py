from theano_object import *

class MyModule(TheanoObject):
    def __init__(self, a=3, b=9):
        super(MyModule, self).__init__()
        self.a = self.symbolic_member(a) #creates a variable for a
        self.b = self.symbolic_member(b) #creates a variable for b

    @symbolic_fn
    def add(self, x):
        return RVal(self.a + self.b + x)

    @symbolic_fn_opts(mode='FAST_COMPILE')
    def sub(self, x):
        outputs = (self.a - x, self.b - x)
        updates = {self.b: self.b-x}
        return RVal(outputs, updates)

    def normal_function(self, x):
        return self.add(x) + self.sub(x)  #use numpy addition

    @symbolic_fn
    def use_submodule(self, x):
        return RVal(self.a + x + self.submodule.b)

def run(TF):
    def deco(f):
        if TF:
            print 'running test', f.__name__
            f()
        return f
    return deco

@run(True)
def test_outputs():
    MM = MyModule(3, 4)
    assert MM.add(5) == 12
    assert MM.b.get() == 4
    MM.sub(3)
    assert MM.b.get() == 1 #test get()
    assert MM.add(5) == 9 #test that b's container is shared between add and sub
    MM.b.set(2) #test set
    assert MM.b.get() == 2 #test get()
    assert MM.add(5) == 10 #test that b's container is shared between add and sub

@run(True)
def test_submodule():
    MM = MyModule(1,2)
    MM.submodule = MyModule(3,4)
    assert MM.add(5) == 8
    MM.submodule.sub(7)
    assert MM.submodule.b.get() == -3
    assert MM.use_submodule(0) == -2 #self.a is 1 + self.submodule.b is -3


@run(False)
def test_misc_prints():
    MM = MyModule()
    print MM
    print 'add', MM.add(4)
    print 'b', MM.value(MM.b)
    print 'sub', MM.sub(45)
    print 'b', MM.value(MM.b)
    print MM.sub(23)
    print MM.add(9)
    print MM.add(19)
    print 'b', MM.value(MM.b)
    print 'a', MM.value(MM.a)
    MM.value_set(MM.a,6)
    MM.value_set(MM.b,6)
    print MM.add(6)

    try:
        MM.b = 5
    except Exception, e:
        print e
    MM.del_member(MM.b)
    try:
        print 'b', MM.value(MM.b)
    except Exception, e:
        print e
    MM.b = 'asdffd'
    try:
        print 'b', MM.value(MM.b)
    except Exception, e:
        print e
    try:
        print 'b', MM.value(MM.b)
    except Exception, e:
        print 'E', e
    print MM.b
    print 'a', MM.value(MM.a)


