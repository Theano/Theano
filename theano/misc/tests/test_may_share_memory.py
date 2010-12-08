import numpy

import theano

try:
    import scipy.sparse
    scipy_imported = True
except ImportError:
    scipy_imported = False

from theano.misc.may_share_memory import may_share_memory

def test_may_share_memory():
    a=numpy.random.rand(5,4)
    b=numpy.random.rand(5,4)
    as_ar = lambda a: theano._asarray(a, dtype='int32')
    for a_,b_,rep in [(a,a,True),(b,b,True),(a,b,False),
                      (a,a[0],True),(a,a[:,0],True),
                      (a,(0,),False),(a,1,False),
                      ]:

        assert may_share_memory(a_,b_,False)==rep
        if rep == False:
            try:
                may_share_memory(a_,b_)
                raise Exception("An error was expected")
            except:
                pass

if scipy_imported:
    def test_may_share_memory_scipy():
        a=scipy.sparse.csc_matrix(scipy.sparse.eye(5,3))
        b=scipy.sparse.csc_matrix(scipy.sparse.eye(4,3))
        as_ar = lambda a: theano._asarray(a, dtype='int32')
        for a_,b_,rep in [(a,a,True),(b,b,True),(a,b,False),
                          (a,a.data,True),(a,a.indptr,True),(a,a.indices,True),(a,as_ar(a.shape),False),
                          (a.data,a,True),(a.indptr,a,True),(a.indices,a,True),(as_ar(a.shape),a,False),
                          (b,b.data,True),(b,b.indptr,True),(b,b.indices,True),(b,as_ar(b.shape),False),
                          (b.data,b,True),(b.indptr,b,True),(b.indices,b,True),(as_ar(b.shape),b,False),
                          (b.data,a,False),(b.indptr,a,False),(b.indices,a,False),(as_ar(b.shape),a,False),
                          ]:

            assert may_share_memory(a_,b_)==rep
            if rep == False:
                try:
                    may_share_memory(a_,b_)
                    raise Exception("An error was expected")
                except:
                    pass
