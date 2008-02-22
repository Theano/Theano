import unittest, os, sys

if __name__ == '__main__':
    suite = None
    filenames = os.listdir('.') + ['gof.'+s for s in os.listdir('gof')]
    for filename in filenames:
        if filename[-3:] == '.py':
            modname = filename[:-3]
            tests = unittest.TestLoader().loadTestsFromModule(__import__(modname))
            if tests.countTestCases() > 0:
                print >>sys.stderr, 'Testing', modname
                if suite is None:
                    suite = tests
                else:
                    suite.addTests(tests)

    unittest.TextTestRunner(verbosity=1).run(suite)

