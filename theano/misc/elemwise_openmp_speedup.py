from __future__ import absolute_import, print_function, division
import os
import subprocess
import sys
from optparse import OptionParser
from locale import getpreferredencoding

import theano
from theano.compat import decode_with

console_encoding = getpreferredencoding()

parser = OptionParser(usage='%prog <options>\n Compute time for'
                      ' fast and slow elemwise operations')
parser.add_option('-N', '--N', action='store', dest='N',
                  default=theano.config.openmp_elemwise_minsize, type="int",
                  help="Number of vector elements")


def runScript(N):
    script = 'elemwise_time_test.py'
    path = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(['python', script, '--script', '-N', str(N)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=path)
    (out, err) = proc.communicate()
    if err:
        print(err)
        sys.exit()
    return list(map(float, decode_with(out, console_encoding).split(" ")))

if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)
    if hasattr(options, "help"):
        print(options.help)
        sys.exit(0)
    orig_flags = os.environ.get('THEANO_FLAGS', '')
    os.environ['THEANO_FLAGS'] = orig_flags + ',openmp=false'
    (cheapTime, costlyTime) = runScript(N=options.N)
    os.environ['THEANO_FLAGS'] = orig_flags + ',openmp=true'
    (cheapTimeOpenmp, costlyTimeOpenmp) = runScript(N=options.N)

    if cheapTime > cheapTimeOpenmp:
        cheapSpeed = cheapTime / cheapTimeOpenmp
        cheapSpeedstring = "speedup"
    else:
        cheapSpeed = cheapTimeOpenmp / cheapTime
        cheapSpeedstring = "slowdown"

    if costlyTime > costlyTimeOpenmp:
        costlySpeed = costlyTime / costlyTimeOpenmp
        costlySpeedstring = "speedup"
    else:
        costlySpeed = costlyTimeOpenmp / costlyTime
        costlySpeedstring = "slowdown"
    print("Timed with vector of %d elements" % options.N)
    print("Fast op time without openmp %fs with openmp %fs %s %2.2f" % (
        cheapTime, cheapTimeOpenmp, cheapSpeedstring, cheapSpeed))

    print("Slow op time without openmp %fs with openmp %fs %s %2.2f" % (
        costlyTime, costlyTimeOpenmp, costlySpeedstring, costlySpeed))
