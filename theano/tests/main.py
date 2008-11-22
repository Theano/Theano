import unittest

def main(modulename):
    if 0:
        unittest.main()
    elif 1:
        module = __import__(modulename)
        tests = unittest.TestLoader().loadTestsFromModule(module)
        tests.debug()
    else:
        testcases = []
        testcases.append(T_function_module)

        #<testsuite boilerplate>
        testloader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for testcase in testcases:
            suite.addTest(testloader.loadTestsFromTestCase(testcase))
        unittest.TextTestRunner(verbosity=2).run(suite)
        #</boilerplate>
