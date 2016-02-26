from theano.sandbox.cuda.cula import gpu_solve
import numpy as np
import theano.tensor as TT
import theano

def thrash():
    import numpy as np

    A_val = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")

    #b_val = np.asarray([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], dtype="float32")
    b_val = np.asarray([[0.5], [0.5], [0.5]], dtype="float32")

    A_empty = np.zeros((3, 3)).astype("float32")
    b_empty = np.zeros((3, 1)).astype("float32")

    import theano
    A = TT.matrix("A", dtype="float32")
    b = TT.matrix("b", dtype="float32")

    #theano.config.compute_test_value = 'warn'
    #A.tag.test_value = A_val
    #b.tag.test_value = b_val

    #A = theano.shared(A_val)
    #b = theano.shared(b_val)
    from theano.misc.pycuda_utils import to_gpuarray

    solver = gpu_solve(A, b)
    fn = theano.function([A, b], [solver])
    res = fn(A_val, b_val)
    print(np.asarray(res[0]))
    #import ipdb; ipdb.set_trace()

def thrash2():
    import numpy as np

    A_val = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    #A_val = np.random.uniform(-0.01, 0.01, (10, 10)).astype("float32")
    #A_val +=1
    #A_val = np.linalg.svd(A_val)[0]
    #A_val = (A_val + A_val.T) / 2.0

    x_val = np.random.uniform(-0.4, 0.4, (A_val.shape[1], 1)).astype("float32")
    b_val = np.dot(A_val, x_val)

    #b_val = np.asarray([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], dtype="float32")
    #b_val = np.asarray([[0.5], [0.5], [0.5]], dtype="float32")

    #A_empty = np.zeros((A_val.shape[1], A_val.shape[1])).astype("float32")
    x_res = np.zeros((A_val.shape[1], 1)).astype("float32")

    import theano
    A = TT.matrix("A", dtype="float32")
    b = TT.matrix("b", dtype="float32")

    #theano.config.compute_test_value = 'warn'
    #A.tag.test_value = A_val
    #b.tag.test_value = b_val

    #A = theano.shared(A_val)
    #b = theano.shared(b_val)

    from theano.misc.pycuda_utils import to_gpuarray

    solver = gpu_solve(A, b)
    fn = theano.function([A, b], [solver])
    res = fn(A_val, b_val)
    res[0].get(x_res)
    print(np.allclose(x_res, x_val))

    import ipdb; ipdb.set_trace()


thrash2()
