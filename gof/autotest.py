import unittest, os, sys, traceback

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



    suite = None
    filenames = os.listdir('.')
    for filename in filenames:
        if filename[-3:] == '.py':
            modname = filename[:-3]
            if modname in ['__init__', 'autotest']: continue
            #print >>sys.stderr, 'Loading', modname

            try:
                module = __import__(modname)
            except Exception, e:
                print >>sys.stderr, "===================================================="
                print >>sys.stderr, "Failed to load module %s" % modname
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
    if debugparam:
        suite.debug()
    else:
        unittest.TextTestRunner(verbosity=1).run(suite)

