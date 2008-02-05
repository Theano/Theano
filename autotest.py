import unittest, os, sys

for filename in os.listdir('.'):
    if filename == __file__: continue
    #continue
    if filename[-3:] == '.py':
        modname = filename[:-3]
        suite = unittest.TestLoader().loadTestsFromModule(__import__(modname))
        #suite.addTests(unittest.TestLoader().loadTestsFromModule(__import__(modname)))
        if suite.countTestCases() > 0:
            print >>sys.stderr, 'Testing', modname, '(%s)'% (filename),
            unittest.TextTestRunner(verbosity=1).run(suite)

