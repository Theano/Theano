import numpy
import unittest
import os

def makeTester(fname):
    
    class Test(unittest.TestCase):
        def test_example(self):
            print 'Executing file', self.fname

    Test.__name__ = fname
    Test.fname = fname 
    return Test

def test_module_doc():
    """
    This test executes all of the Module code examples.
    It goes through the directory and executes all .py files.
    """

    for fname in os.listdir('.'):
        if fname.endswith('.py'):
            f = fname.split('.')[0]
            print 'Executing ', fname
            execfile(fname, locals())
    
