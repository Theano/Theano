import os
import subprocess
import sys


def runScript():
    script = 'elemwise_time_test.py'
    dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.Popen(['python', script, '--script'], stdout=subprocess.PIPE, stderr = subprocess.PIPE, cwd = dir)
    (out, err) = proc.communicate()
    if err:
        print err
        sys.exit()
    return map(float, out.split(" "))

if __name__ == '__main__':
    (cheapTime, costlyTime) = runScript()
    os.environ['THEANO_FLAGS'] = 'openmp=true'
    (cheapTimeOpenmp, costlyTimeOpenmp) = runScript()

    if cheapTime > cheapTimeOpenmp:
        cheapSpeed = (cheapTime - cheapTimeOpenmp) / cheapTime
        cheapSpeedstring = "speedup"
    else:
        cheapSpeed = (cheapTimeOpenmp - cheapTime) / cheapTimeOpenmp
        cheapSpeedstring = "slowdown"

    if cheapTime > cheapTimeOpenmp:
        costlySpeed = (costlyTime - costlyTimeOpenmp) / costlyTime
        costlySpeedstring = "speedup"
    else:
        costlySpeed = (costlyTimeOpenmp - costlyTime) / costlyTimeOpenmp
        costlySpeedstring = "slowdown"

    print "Cheap op time without openmp %fs with openmp %fs %s %2.2f%%" % (cheapTime, cheapTimeOpenmp, cheapSpeedstring, cheapSpeed*100)
    
    print "Costly op time without openmp %fs with openmp %fs %s %2.2f%%" % (costlyTime, costlyTimeOpenmp, costlySpeedstring, costlySpeed*100)
