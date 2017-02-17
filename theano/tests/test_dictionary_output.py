from __future__ import absolute_import, print_function, division
import unittest
import theano
import theano.tensor as T


class dictionary_output_checker(unittest.TestCase):

    def test_output_dictionary(self):

        '''
        Tests that theano.function works when outputs is a dictionary
        '''

        x = T.scalar()
        f = theano.function([x], outputs={'a': x, 'c': x * 2,
                                          'b': x * 3, '1': x * 4})

        outputs = f(10.0)

        assert outputs['a'] == 10.0
        assert outputs['b'] == 30.0
        assert outputs['1'] == 40.0
        assert outputs['c'] == 20.0

    def test_input_named_variables(self):

        '''
        Tests that named variables work when outputs is a dictionary
        '''

        x = T.scalar('x')
        y = T.scalar('y')

        f = theano.function([x, y], outputs={'a': x + y, 'b': x * y})

        assert f(2, 4) == {'a': 6, 'b': 8}
        assert f(2, y=4) == f(2, 4)
        assert f(x=2, y=4) == f(2, 4)

    def test_output_order_sorted(self):

        '''
        Tests that the output keys are sorted correctly.
        '''

        x = T.scalar('x')
        y = T.scalar('y')
        z = T.scalar('z')
        e1 = T.scalar('1')
        e2 = T.scalar('2')

        f = theano.function([x, y, z, e1, e2], outputs={'x': x, 'y': y, 'z': z,
                                                        '1': e1, '2': e2})

        assert '1' in str(f.outputs[0])
        assert '2' in str(f.outputs[1])
        assert 'x' in str(f.outputs[2])
        assert 'y' in str(f.outputs[3])
        assert 'z' in str(f.outputs[4])

    def test_composing_function(self):

        '''
        Tests that one can compose two theano functions when the outputs are
        provided in a dictionary.
        '''

        x = T.scalar('x')
        y = T.scalar('y')

        a = x + y
        b = x * y

        f = theano.function([x, y], outputs={'a': a, 'b': b})

        a = T.scalar('a')
        b = T.scalar('b')

        l = a + b
        r = a * b

        g = theano.function([a, b], outputs=[l, r])

        result = g(**f(5, 7))

        assert result[0] == 47.0
        assert result[1] == 420.0

    def test_output_list_still_works(self):

        '''
        Test that theano.function works if outputs is a list.
        '''

        x = T.scalar('x')

        f = theano.function([x], outputs=[x * 3, x * 2, x * 4, x])

        result = f(5.0)

        assert result[0] == 15.0
        assert result[1] == 10.0
        assert result[2] == 20.0
        assert result[3] == 5.0

    def test_debug_mode_dict(self):

        '''
        Tests that debug mode works where outputs is a dictionary.
        '''

        x = T.scalar('x')

        f = theano.function([x], outputs={'1': x, '2': 2 * x,
                                          '3': 3 * x}, mode="DEBUG_MODE")

        result = f(3.0)

        assert result['1'] == 3.0
        assert result['2'] == 6.0
        assert result['3'] == 9.0

    def test_debug_mode_list(self):

        '''
        Tests that debug mode works where the outputs argument is a list.
        '''

        x = T.scalar('x')

        f = theano.function([x], outputs=[x, 2 * x, 3 * x], mode="DEBUG_MODE")

        result = f(5.0)

        assert result[0] == 5.0
        assert result[1] == 10.0
        assert result[2] == 15.0

    def test_key_string_requirement(self):
        '''
        Tests that an exception is thrown if a non-string key is used in
        the outputs dictionary.
        '''
        x = T.scalar('x')
        try:
            theano.function([x], outputs={1.0: x})
            raise Exception("Did not throw exception with 1.0 as only key")
        except AssertionError:
            pass
        try:
            theano.function([x], outputs={1.0: x, "a": x**2})
            raise Exception("Did not throw exception with 1.0 as one key")
        except AssertionError:
            pass
        try:
            theano.function([x], outputs={(1, "b"): x, 1.0: x**2})
            raise Exception("Did not throw exception with tuple as key")
        except AssertionError:
            pass
