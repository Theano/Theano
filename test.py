
import unittest, os, sys, traceback, commands


def test_module(module_path, debugmode = False):
    files = commands.getoutput("find %s -name test_*.py" % module_path)
    suite = None
    for file in files.split("\n"):
        try:
            module = __import__(file[:-3])
        except Exception, e:
            print >>sys.stderr, "===================================================="
            print >>sys.stderr, "Failed to load %s" % file
            print >>sys.stderr, "===================================================="
            traceback.print_exc()
            print >>sys.stderr, "===================================================="
            continue
        
        tests = unittest.TestLoader().loadTestsFromModule(module)
        if tests.countTestCases() > 0:
            print >>sys.stderr, 'Testing', file
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

    test_module("theano")


