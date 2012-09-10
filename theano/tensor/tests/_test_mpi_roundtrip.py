from mpi4py import MPI
comm = MPI.COMM_WORLD
import theano
from theano.tensor.io import send, recv
import numpy as np
from sys import stdout


rank = comm.Get_rank()
size = comm.Get_size()
print size
print rank

shape = (10, 10)
dtype = 'float32'

if rank == 0:
    x = theano.tensor.matrix('x', dtype=dtype)
    y = x + 1
    send_request = send(x, 1, 11)

    z = recv(shape, dtype, 1, 12)

    f = theano.function([x], [send_request, z])

    xx = np.random.rand(*shape).astype(dtype)

    zz = f(xx)

    same = np.linalg.norm(zz - (xx+1)*2) < .001
    stdout.write(str(same))

if rank == 1:

    y = recv(shape, dtype, 0, 11)
    z = y * 2
    send_request = send(z, 0, 12)

    f = theano.function([], send_request)

    f()
