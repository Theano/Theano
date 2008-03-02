
import unittest
from constructor import *
import random

class MyAllocator(Allocator):

    def __init__(self, fn):
        self.fn = fn

    def __call__(self):
        return self.fn.__name__

def f1(a, b, c):
    return a + b + c

def f2(x):
    return "!!%s" % x


class _test_Constructor(unittest.TestCase):

    def test_0(self):
        c = Constructor(MyAllocator)
        c.update({"fifi": f1, "loulou": f2})
        assert c.fifi() == 'f1' and c.loulou() == 'f2'

    def test_1(self):
        c = Constructor(MyAllocator)
        c.add_module(random)
        assert c.random.random() == 'random' and c.random.randint() == 'randint'

    def test_2(self):
        c = Constructor(MyAllocator)
        c.update({"fifi": f1, "loulou": f2})
        globals().update(c)
        assert fifi() == 'f1' and loulou() == 'f2'


if __name__ == '__main__':
    unittest.main()





