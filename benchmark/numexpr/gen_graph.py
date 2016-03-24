from __future__ import absolute_import, print_function, division
import numpy as np
import numexpr as ne
import timeit
import theano
import theano.tensor as T
try:
    import pylab
    import matplotlib.pyplot  as pyplot
except ImportError:
    pass

def timeit_2vector_theano(init, nb_element=1e6, nb_repeat=3, nb_call=int(1e2), expr="a**2 + b**2 + 2*a*b"):
    t3 = timeit.Timer("tf(av,bv)",
                      """
import theano
import theano.tensor as T
import numexpr as ne
from theano.tensor import exp
%(init)s
av=a
bv=b
a=T.dvector()
b=T.dvector()
tf= theano.function([a,b],%(expr)s)
"""%locals()
)
    ret=t3.repeat(nb_repeat,nb_call)
    return np.asarray(ret)


def timeit_2vector(nb_element=1e6, nb_repeat=3, nb_call=int(1e2), expr="a**2 + b**2 + 2*a*b", do_unalign=False, do_amd=True):
    """Returns a dictionary whose keys are implementations ('numpy', 'numexpr', 'theano', etc.)
    and whose values are numpy arrays of times taken to evalute the given problem.
    """
    rval = dict() 
    print()
    print("timeit_2vector(nb_element=%(nb_element)s,nb_repeat=%(nb_repeat)s,nb_call=%(nb_call)s, expr=%(expr)s, do_unalign=%(do_unalign)s)"%locals())

    if do_unalign:
        init = "import numpy as np; a = np.empty(%(nb_element)s, dtype='b1,f8')['f1'];b = np.empty(%(nb_element)s, dtype='b1,f8')['f1'];a[:] = np.arange(len(a));b[:] = np.arange(len(b));"%locals()
    else:
        init = "import numpy as np; a = np.arange(%(nb_element)s);b = np.arange(%(nb_element)s)"%locals()
    t1 = timeit.Timer("%(expr)s"%locals(),"from numpy import exp; %(init)s"%locals())
    numpy_times = np.asarray(t1.repeat(nb_repeat,nb_call))
    print("NumPy time: each time=",numpy_times, "min_time=", numpy_times.min())
    rval['numpy'] = numpy_times

    t2 = timeit.Timer("""ne.evaluate("%(expr)s")"""%locals(),
                      "import numexpr as ne; %(init)s"%locals())
    numexpr_times=np.asarray(t2.repeat(nb_repeat,nb_call))
    rval['numexpr'] = numexpr_times
    print("Numexpr time: each time=",numexpr_times,'min_time=', numexpr_times.min())

    theano.config.lib.amdlibm = False
    theano_times = timeit_2vector_theano(init, nb_element,nb_repeat,nb_call,expr)
    print("Theano time: each time=",theano_times, 'min_time=',theano_times.min())
    rval['theano'] = theano_times

    if do_amd:
        theano.config.lib.amdlibm = True
        theanoamd_times = timeit_2vector_theano(init, nb_element,nb_repeat,nb_call,expr)
        print("Theano+amdlibm time",theanoamd_times, theanoamd_times.min())
        rval['theano_amd'] = theanoamd_times

    print("time(NumPy) /  time(numexpr) = ",numpy_times.min()/numexpr_times.min())
    print("time(NumPy) / time(Theano)",numpy_times.min()/theano_times.min())
    print("time(numexpr) / time(Theano)",numexpr_times.min()/theano_times.min())
    if do_amd:
        print("time(NumPy) / time(Theano+amdlibm)",numpy_times.min()/theanoamd_times.min())
        print("time(numexpr) / time(Theano+amdlibm)",numexpr_times.min()/theanoamd_times.min())
    return rval

def exec_timeit_2vector(expr, nb_call_scal=1, fname=None, do_unalign=False, do_amd=True):
    #exp = [(1,100000),(1e1,100000),(1e2,100000),(1e3,100000), (5e3,50000),
    exp = [(1e3,100000),(5e3,50000), \
           (1e4,10000),(5e4,5000),(1e5,2000),(1e6,200),(1e7,10)
           ]
    exp = [(1e3,100000),(5e3,50000)]
    runtimes=[]

    for nb_e, nb_c in exp:
        runtimes.append(timeit_2vector(nb_element=nb_e, nb_repeat=3, nb_call=nb_c*nb_call_scal, expr=expr, do_amd=do_amd))
    if do_unalign:
        runtimes_unalign=[]
        for nb_e, nb_c in exp:
            runtimes_unalign.append(timeit_2vector(nb_element=nb_e, nb_repeat=3, nb_call=nb_c*nb_call_scal, expr=expr, do_unalign=True, do_amd=do_amd))

    print('Runtimes list = ', runtimes)
    numexpr_speedup = np.asarray([t['numpy'].min()/t['numexpr'].min() for t in runtimes],"float32")
    print("time(NumPy) / time(numexpr)", end=' ')
    print(numexpr_speedup, numexpr_speedup.min(), numexpr_speedup.max())

    theano_speedup = np.asarray([t['numpy'].min()/t['theano'].min() for t in runtimes],"float32")
    print("time(NumPy) / time(Theano)", end=' ')
    print(theano_speedup, theano_speedup.min(), theano_speedup.max())

    theano_numexpr_speedup = np.asarray([t['numexpr'].min()/t['theano'].min() for t in runtimes],"float32")
    print("time(numexpr) / time(Theano)", end=' ')
    print(theano_numexpr_speedup, theano_numexpr_speedup.min(), theano_numexpr_speedup.max())

    if do_amd:
        theano_speedup2 = np.asarray([t['numpy'].min()/t['theano_amd'].min() for t in runtimes],"float32")
        print("time(NumPy) / time(theano+amdlibm)", end=' ')
        print(theano_speedup,theano_speedup.min(),theano_speedup.max())

        theano_numexpr_speedup2 = np.asarray([t['numexpr'].min()/t['theano_amd'].min() for t in runtimes],"float32")
        print("time(numexpr) / time(theano+amdlibm)", end=' ')
        print(theano_numexpr_speedup, theano_numexpr_speedup.min(), theano_numexpr_speedup.max())

    if 'pylab' not in globals():
        return

    nb_calls=[e[0] for e in exp]
    for cmp in range(1,len(time[0])):
        speedup = np.asarray([t[0].min()/t[cmp].min() for t in time],"float32")
        pylab.semilogx(nb_calls, speedup, linewidth=1.0)
    if do_unalign:
        for cmp in range(1,len(time[0])):
            speedup = np.asarray([t[0].min()/t[cmp].min() for t in time_unalign],"float32")
            pylab.semilogx(nb_calls, speedup, linewidth=1.0)

    pylab.axhline(y=1, linewidth=1.0, color='black')
        
    pylab.xlabel('Dimension of real valued vectors a and b')
    pylab.ylabel('Speed up vs NumPy')
    if do_unalign and do_amd:
        pylab.legend(("Numexpr","Theano","Theano(amdlibm)", "Numexpr(unalign)",
                      "Theano(unalign)","Theano(amdlibm,unalign)"),loc='upper left')
    elif do_unalign and not do_amd:
        pylab.legend(("Numexpr","Theano","Numexpr(unalign)",
                      "Theano(unalign)",),loc='upper left')
    elif not do_unalign and do_amd:
        pylab.legend(("Numexpr","Theano","Theano(amdlibm)"),loc='upper left')
    else:
        pylab.legend(("Numexpr","Theano"),loc='upper left')

    pylab.grid(True)
    if fname:
        pylab.savefig(fname)
        pylab.clf()
    else:
        pylab.show()

def execs_timeit_2vector(exprs, fname=None):
    """
    exprs is a list of list of expr to evaluate
    The first level of list is put into different graph section in the same graph.
    The second level is the expression to put in each section
    """
    #exp = [(1,100000),(1e1,100000),(1e2,100000),(1e3,100000), (5e3,50000),
    exp = [(1e3,100000),(5e3,50000), \
           (1e4,10000),(5e4,5000),(1e5,2000),(1e6,200),(1e7,10)
           ]
    ### TO TEST UNCOMMENT THIS LINE
    # exp = [(1,1000),(1e1,1000),(1e2,1000),]
    times=[]
    str_expr=[]
    for g_exprs in exprs:
        for expr in g_exprs:
            nb_call_scal=1
            if isinstance(expr,tuple):
                nb_call_scal=expr[1]
                expr = expr[0]
            str_expr.append(expr)
            time=[]
            for nb_e, nb_c in exp:
                time.append(timeit_2vector(nb_element=nb_e, nb_repeat=3, nb_call=nb_c*nb_call_scal, expr=expr, do_amd=False))
            times.append(time)
    if 'pylab' not in globals():
        return

    nb_calls=[e[0] for e in exp]
    legends=[]
    colors=['b','r','g','c', 'm', 'y']
    assert len(colors)>=len(times)
    fig = pylab.figure()
    for idx,(time,expr) in enumerate(zip(times,str_expr)):

        ###
        ###
        ###
        # Creating each subplot
        ###
        ###
        ###
        ###
        pylab.subplot(220+idx+1)
        pylab.subplots_adjust(wspace=0.25, hspace=0.25)
        #legend=[]
        #plot = fig.add_subplot(1,len(exprs),idx)
        speedup = [t["numpy"].min()/t["numexpr"].min() for t in time]

        pylab.semilogx(nb_calls, speedup, linewidth=1.0,  color='r')
        speedup = [t["numpy"].min()/t["theano"].min() for t in time]
        pylab.semilogx(nb_calls, speedup, linewidth=1.0, color = 'b')
        pylab.grid(True)
        if (idx == 2) or (idx == 3):
            pylab.xlabel('Dimension of vectors a and b', fontsize = 15)
        if (idx == 0) or (idx == 2):
            pylab.ylabel('Speed up vs NumPy', fontsize = 15)
        pylab.axhline(y=1, linewidth=1.0, color='black')
        pylab.xlim(1e3,1e7)
        pylab.xticks([1e3,1e5,1e7],['1e3','1e5','1e7'])
        pylab.title(expr)



    if fname:
        fig.savefig(fname)
        pylab.clf()
    else:
        pylab.show()
    
execs_timeit_2vector([
        ["a**2 + b**2 + 2*a*b",
         "2*a + 3*b",
         "a+1",],
        [("2*a + b**10",.2)]
#"2*a + b*b*b*b*b*b*b*b*b*b",
#("2*a + exp(b)",.3),
],fname="multiple_graph.pdf"
)
###
### This case is the one gived on numexpr web site(http://code.google.com/p/numexpr/) as of 16 June 2010
### a**2 + b**2 + 2*a*b
#exec_timeit_2vector("a**2 + b**2 + 2*a*b",fname="speedup_numexpr_mulpow2vec.png", do_amd=False)

###
### This case is the one gived on numexpr web site(http://code.google.com/p/numexpr/wiki/Overview) as of 16 June 2010
### 2*a + 3*b
#exec_timeit_2vector("2*a + 3*b",fname="speedup_numexpr_mul2vec.png", do_amd=False)

###
### This case is the one gived on numexpr web site(http://code.google.com/p/numexpr/wiki/Overview) as of 16 June 2010
### 2*a + b**10
#exec_timeit_2vector("2*a + b**10",.2,fname="speedup_numexpr_mulpow2vec_simple.png")
#exec_timeit_2vector("2*a + b*b*b*b*b*b*b*b*b*b",fname="speedup_numexpr_mulpow2vec_simpleV2.png", do_amd=False)

###
### We try to see if the pow optimized speed is available for exp too.
### 2*a + exp(b)
#exec_timeit_2vector("2*a + exp(b)",.3,fname="speedup_numexpr_mulexp2vec.png")

###
### The simplest case where we should show the overhead at its maximum effect
### a+1
#exec_timeit_2vector("a+1",fname="speedup_numexpr_add1vec.png")


#exec_timeit_2vector("a+1",.2,fname="speedup_numexpr_add1vec_unalign.png",do_unalign=True, do_amd=False)
#exec_timeit_2vector("2*a + b**10",.1,fname="speedup_numexpr_mulpow2vec_simple_unalign.png",do_unalign=True)
