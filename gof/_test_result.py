
import unittest
from result import *


class Double(ResultBase):

    def __init__(self, data, name = "oignon"):
        ResultBase.__init__(self, role = None, name = name)
        assert isinstance(data, float)
        self.data = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __copy__(self):
        return Double(self.data, self.name)

class MyResult(ResultBase):

    def __init__(self, name):
        ResultBase.__init__(self, role = None, name = name)
        self.data = [1000]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __copy__(self):
        return MyResult(self.name)


class _test_ResultBase(unittest.TestCase):
    def test_0(self):
        r = ResultBase()
    def test_1(self):
        r = ResultBase()
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
