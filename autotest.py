import unittest, os, sys

def test_root_dir():
    suite = None
    filenames = os.listdir('.')
    for filename in filenames:
        if filename[-3:] == '.py' and filename[0:5] == '_test':
            #print >>sys.stderr, 'Loading', modname
            modname = filename[0:-3]
            tests = unittest.TestLoader().loadTestsFromModule(__import__(modname))
            if tests.countTestCases() > 0:
                print >>sys.stderr, 'Testing', modname
                if suite is None:
                    suite = tests
                else:
                    suite.addTests(tests)

    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == '__main__':
    try:
        os.system('cd gof; %s autotest.py' % sys.argv[1])
    except IndexError, e:
        os.system('cd gof; python autotest.py')
    test_root_dir()

