from theano.tensor.io import send, recv
import theano
import subprocess
import os

def test_recv():
    x = recv((10,10), 'float64', 0, 11)
    assert x.dtype == 'float64'
    assert x.broadcastable == (False, False)

    recvnode = x.owner.inputs[0].owner
    assert recvnode.op.source == 0
    assert recvnode.op.tag    == 11

def test_send():
    x = theano.tensor.matrix('x')
    y = send(x, 1, 11)
    sendnode = y.owner.inputs[0].owner
    assert sendnode.op.dest == 1
    assert sendnode.op.tag  == 11

def test_can_make_function():
    x = recv((5,5), 'float32', 0, 11)
    y = x+1
    assert theano.function([], [y])

def test_mpi_roundtrip():
#    p = subprocess.Popen(executable="mpiexec",
#                         args = ("-np", "2",
#                                 "python",
#                                 "theano/tensor/tests/_test_mpi_roundtrip.py"),
#                         stdout=subprocess.PIPE)
#    assert p.stdout.read() == "True"
    result = os.popen("mpiexec -np 2 python "
                      "theano/tensor/tests/_test_mpi_roundtrip.py").read()
    assert result == "True"
