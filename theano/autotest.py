import unittest, os, sys, traceback

def test_root_dir(debugmode=False):
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
    if debugmode:
        suite.debug()
    else:
        unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == '__main__':

    def printUsage():
        print >>sys.stderr, "Bad argument: ",sys.argv
        print >>sys.stderr, "only --debug is supported"
        sys.exit(1)
    debugparam=""

    if len(sys.argv)==2:
        if sys.argv[1]=="--debug":
            debugparam="--debug"
            sys.argv.remove(debugparam)
        else:
            printUsage()
    elif len(sys.argv)>2:
        printUsage()

    if len(sys.argv) >= 2:
        cmd = sys.argv[1]
    else:
        cmd = 'python'

    for dir in 'gof compile scalar tensor sparse'.split():
        os.system('cd %s; %s autotest.py %s' % (dir, cmd, debugparam))
    test_root_dir(debugparam!="")


