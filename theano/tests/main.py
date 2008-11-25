import unittest,sys

def main(modulename):
    debug = False

    if 0:
        unittest.main()
    elif len(sys.argv)==2 and sys.argv[1]=="--debug":
        module = __import__(modulename)
        tests = unittest.TestLoader().loadTestsFromModule(module)
        tests.debug()
    elif len(sys.argv)==1:
        module = __import__(modulename)
        tests = unittest.TestLoader().loadTestsFromModule(module)
        unittest.TextTestRunner(verbosity=2).run(tests)
    else:
        print "options: [--debug]"
