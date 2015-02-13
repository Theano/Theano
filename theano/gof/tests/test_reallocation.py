"Test of reduce allocation"

import unittest

import theano
import theano.tensor as T


class Test_reallocation(unittest.TestCase):

    """
    Test of Theano reallocation
    """

    def test_reallocation(self):

        pre_config = theano.config.allow_gc

        try:
            theano.config.allow_gc = False

            x = T.scalar('x')
            y = T.scalar('y')

            z = T.tanh(x + y) + T.cosh(x + y)

            if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
                m = "FAST_RUN"
            else:
                m = None

            m = theano.compile.get_mode(m).excluding('fusion', 'inplace')

            f = theano.function([x, y], z, name="test_reduce_memory",
                                mode=m)

            output = f(1, 2)
            storage_map = f.fn.storage_map

            def check_storage(storage_map):
                for i in storage_map.keys():
                    keys_copy = storage_map.keys()[:]
                    keys_copy.remove(i)
                    for o in keys_copy:
                        if storage_map[i][0] == storage_map[o][0]:
                            return True
                return False

            assert check_storage(storage_map)

        finally:
            theano.config.allow_gc = pre_config

if __name__ == "__main__":
    unittest.main()
