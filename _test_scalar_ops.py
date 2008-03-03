
import unittest

from gof import ResultBase, Op, Env, modes


from scalar_ops import *


def inputs():
    x = modes.BuildEvalMode(as_scalar(1.0, 'x'))
    y = modes.BuildEvalMode(as_scalar(2.0, 'y'))
    z = modes.BuildEvalMode(as_scalar(3.0, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
    inputs = [input.r for input in inputs]
    outputs = [output.r for output in outputs]
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_ScalarOps(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        assert e.r.data == 1.5


if __name__ == '__main__':
    unittest.main()




