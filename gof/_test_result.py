
import unittest
from result import *

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
