import cPickle
import os
import shutil
import tempfile

import numpy

import theano


def test_function_dump():
    v = theano.tensor.vector()
    fct1 = theano.function([v], v + 1)

    try:
        tmpdir = tempfile.mkdtemp()
        fname = os.path.join(tmpdir, 'test_function_dump.pkl')
        theano.function_dump(fname, [v], v + 1)
        f = open(fname, 'rb')
        l = cPickle.load(f)
        f.close()
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

    fct2 = theano.function(**l)
    x = [1, 2, 3]
    assert numpy.allclose(fct1(x), fct2(x))
