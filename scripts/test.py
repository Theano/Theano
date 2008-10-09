
import unittest, os, sys, traceback, commands
theano_path = os.path.realpath("%s/.." % sys.path[0])
sys.path[0:0] = [theano_path]

def test_module(module_path, debugmode = False):
    files = commands.getoutput("find %s -name test_*.py" % module_path)
    suite = None
    tocut = len("/".join(module_path.split("/")[:-1])) + 1
    for file in files.split("\n"):
        file = file[tocut:]
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
    if suite is None:
        return
    if debugmode:
        suite.debug()
    else:
        unittest.TextTestRunner(verbosity=1).run(suite)

def py_test(module_path):
    py.test.cmdline.main([module_path])

def nopy_test(module_path):
    print >>sys.stderr, "py.test is not installed!"
    print >>sys.stderr, "  easy_install py"
    print >>sys.stderr, "or if you are installing locally"
    print >>sys.stderr, "  easy_install --prefix=/some/local/dir py"
    return None

    files = commands.getoutput("find %s -name test_*.py" % module_path)
    suite = None
    tocut = len("/".join(module_path.split("/")[:-1])) + 1
    for file in files.split("\n"):
        file = file[tocut:]
        try:
            module = __import__(file[:-3])
        except Exception, e:
            print >>sys.stderr, "===================================================="
            print >>sys.stderr, "Failed to load %s" % file
            print >>sys.stderr, "===================================================="
            traceback.print_exc()
            print >>sys.stderr, "===================================================="
            continue
    for method in dir(module):
        if method.startswith("test_"):
            method = getattr(module, method)
            try:
                r = method()
            except Exception, e:
                print >>sys.stderr, "===================================================="
                print >>sys.stderr, "Exception in %s.%s" % (file, method.__name__)
                print >>sys.stderr, "===================================================="
                traceback.print_exc()
                print >>sys.stderr, "===================================================="
            if hasattr(r, 'next'):
                for fargs in r:
                    try:
                        fargs[0](*fargs[1:])
                    except Exception, e:
                        print >>sys.stderr, "===================================================="
                        print >>sys.stderr, "Exception in %s.%s, %s%s" % (file, method.__name__, fargs[0], fargs[1:])
                        print >>sys.stderr, "===================================================="
                        traceback.print_exc()
                        print >>sys.stderr, "===================================================="


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

    mname = os.path.join(theano_path, "theano")
    test_module(mname)
    try:
        import py.test
        py_test(mname)
    except ImportError:
        nopy_test(mname)


