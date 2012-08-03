"""
This file defines Theano flags which need to be defined late in import order.

This is needed as they rely on the values of other previously-defined flags.
"""

import os
import logging
import subprocess
import tempfile

import theano
from theano.configparser import (
        AddConfigVar, BoolParam, ConfigParam, EnumStr, IntParam,
        TheanoConfigParser)
from theano.misc.cpucount import cpuCount

_logger = logging.getLogger('theano.configdefaults_late')

config = TheanoConfigParser()

#http://pyprocessing.berlios.de/
#True if the environment variable (OMP_NUM_THREADS!=1 or
#if we detect more then 1 CPU core) and g++ support OpenMP
#Otherwise False.
default_openmp = True

#Test if the env variable is set
var = os.getenv('OMP_NUM_THREADS', None)
if var:
    try:
        int(var)
    except ValueError:
        raise TypeError("The environment variable OMP_NUM_THREADS"
                        " should be a number, got '%s'." % var)
    else:
        default_openmp = not int(var) == 1
else:
    #Check the number of cores availables.
    count = cpuCount()
    if count == -1:
        _logger.warning("We are not able to detect the number of CPU cores."
                        " We disable openmp by default. To remove this"
                        " warning, set the environment variable"
                        " OMP_NUM_THREADS to the number of threads you"
                        " want theano to use.")
    default_openmp = count > 1

dummy_stdin = open(os.devnull)

if default_openmp and theano.configdefaults.gxx_avail:
    #check if g++ supports openmp. We need to compile a file as the EPD
    #version has openmp enabled in the specs file but does not include
    #the OpenMP files.
    try:
        code = """
        #include <omp.h>
int main( int argc, const char* argv[] )
{
        int res[10];

        for(int i=0; i < 10; i++){
            res[i] = i;
        }
}
        """
        fd, path = tempfile.mkstemp(suffix='.c', prefix='test_omp_')
        try:
            os.write(fd, code)
            os.close(fd)
            fd = None
            proc = subprocess.Popen(['g++', '-fopenmp', path],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    stdin=dummy_stdin.fileno())
            proc.wait()
            if proc.returncode != 0:
                default_openmp = False
        finally:
            # Ensure `fd` is closed before we remove the temporary file.
            try:
                if fd is not None:
                    os.close(fd)
            finally:
                os.remove(path)
    except OSError, e:
        default_openmp = False


del dummy_stdin


AddConfigVar('openmp',
             "Enable (or not) parallel computation on the CPU with OpenMP. "
             "This is the default value used when creating an Op that "
             "supports OpenMP parallelization. It is preferable to define it "
             "via the Theano configuration file ~/.theanorc or with the "
             "environment variable THEANO_FLAGS. Parallelization is only "
             "done for some operations that implement it, and even for "
             "operations that implement parallelism, each operation is free "
             "to respect this flag or not.",
             BoolParam(default_openmp),
             in_c_key=False,
         )
