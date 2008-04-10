import unittest, os, sys, traceback

def test_root_dir():
    suite = None
    filenames = os.listdir('.')
    for filename in filenames:
        if filename[-3:] == '.py' and filename[0:5] == '_test':
            #print >>sys.stderr, 'Loading', modname
            modname = filename[0:-3]

            try:
                module = __import__(modname)
            except Exception, e:
                print >>sys.stderr, "===================================================="
                print >>sys.stderr, "Failed to load %s.py" % modname
                print >>sys.stderr, "===================================================="
                traceback.print_exc()
                print >>sys.stderr, "===================================================="
                continue
                
            tests = unittest.TestLoader().loadTestsFromModule(module)
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

