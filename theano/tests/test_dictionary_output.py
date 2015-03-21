
import unittest
from theano.tests  import unittest_tools as utt
import theano
import theano.tensor as T


class dictionary_output_checker(unittest.TestCase):


    def test_output_list(self): 

        x = T.scalar()

        f = theano.function([x], outputs = [x, x*2, x*3])

        outputs = f(10.0)

        assert outputs[0] == 10.0
        assert outputs[1] == 20.0
        assert outputs[2] == 30.0

    def check_output_dictionary(self): 
        
        x = T.scalar()

        f = theano.function([x], outputs = {'a' : x, 'c' : x*2, 'b' : x*3, 1 : x*4})

        outputs = f(10.0)

        assert outputs['a'] == 10.0
        assert outputs['b'] == 30.0
        assert outputs[1] == 40.0
        assert outputs['c'] == 20.0

    def check_input_dictionary(self): 
        x = T.scalar()
        y = T.scalar()

        f = theano.function([x,y], outputs = {'a' : x + y, 'b' : x * y})

        assert f(2,4) == {'a' : 6, 'b' : 8}
        assert f(2, y = 4) == f(2,4)
        assert f(x = 2, y = 4) == f(2,4)
