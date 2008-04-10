
import unittest
from result import *


class Double(Result):

    def __init__(self, data, name = "oignon"):
        Result.__init__(self, role = None, name = name)
        assert isinstance(data, float)
        self.data = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __copy__(self):
        return Double(self.data, self.name)

class MyResult(Result):

    def __init__(self, name):
        Result.__init__(self, role = None, name = name)
        self.data = [1000]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __copy__(self):
        return MyResult(self.name)


class _test_Result(unittest.TestCase):
    def test_trivial(self):
        r = Result()
        
    def test_state(self):
        r = Result()
        assert r.state is Empty

        r.data = 0
        assert r.data == 0
        assert r.state is Computed
        
        r.data = 1
        assert r.data == 1
        assert r.state is Computed

        r.data = None
        assert r.data == None
        assert r.state is Empty


if __name__ == '__main__':
    unittest.main()
