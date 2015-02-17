"Test of reduce allocation"

import unittest

import theano
import theano.tensor as T


class Test_reallocation(unittest.TestCase):

    """
    Test of Theano reallocation
    """

    def test_reallocation(self):

        x = T.scalar('x')
        y = T.scalar('y')

        z = T.tanh(3*x + y) + T.cosh(x + 5*y)

        m = theano.compile.get_mode(theano.Mode(linker='vm_nogc'))
        m = m.excluding('fusion', 'inplace')

        f = theano.function([x, y], z, name="test_reduce_memory",
                            mode=m)

        output = f(1, 2)
        storage_map = f.fn.storage_map

        def check_storage(storage_map):
            from theano.tensor.var import TensorConstant
            for i in storage_map.keys(): 
                if not isinstance(i, TensorConstant):
                    keys_copy = storage_map.keys()[:]
                    keys_copy.remove(i)
                    for o in keys_copy:
                        if storage_map[i][0] and storage_map[i][0] == storage_map[o][0]:
                            return [True, storage_map[o][0]]
            return [False, None]

        assert check_storage(storage_map)[0]

if __name__ == "__main__":
    unittest.main()
