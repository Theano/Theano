

def make_declare(loop_orders, dtypes, sub):
    decl = ""
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes)):
        var = sub['lv%i' % i]
        decl += """
        %(dtype)s* %(var)s_iter;
        int %(var)s_nd;
        """ % locals()
        for j, value in enumerate(loop_order):
            if value != 'x':
                decl += """
                int %(var)s_n%(value)i;
                int %(var)s_stride%(value)i;
                int %(var)s_jump%(value)i_%(j)i;
                """ % locals()
            else:
                decl += """
                int %(var)s_jump%(value)s_%(j)i;
                """ % locals()
                
    return decl


def make_checks(loop_orders, dtypes, sub):
    init = ""
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes)):
        var = "%%(lv%i)s" % i
        nonx = [x for x in loop_order if x != 'x']
        if nonx:
            min_nd = max(nonx)
            init += """
            if (%(var)s->nd < %(min_nd)s) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                %%(fail)s
            }
            """ % locals()
        adjust = []
        for j, index in reversed([aaa for aaa in enumerate(loop_order)]):
            if index != 'x':
                jump = " - ".join(["%(var)s_stride%(index)s" % locals()] + adjust)
                init += """
                %(var)s_n%(index)s = %(var)s->dimensions[%(index)s];
                %(var)s_stride%(index)s = %(var)s->strides[%(index)s] / sizeof(%(dtype)s);
                %(var)s_jump%(index)s_%(j)s = %(jump)s;
                //printf("%(var)s_jump%(index)s_%(j)s is:");
                //std::cout << %(var)s_jump%(index)s_%(j)s << std::endl;
                """ % locals()
                adjust = ["%(var)s_n%(index)s*%(var)s_stride%(index)s" % locals()]
            else:
                jump = " - ".join(["0"] + adjust)
                init += """
                %(var)s_jump%(index)s_%(j)s = %(jump)s;
                //printf("%(var)s_jump%(index)s_%(j)s is:");
                //std::cout << %(var)s_jump%(index)s_%(j)s << std::endl;
                """ % locals()
                adjust = []
    check = ""
    for matches in zip(*loop_orders):
        to_compare = [(j, x) for j, x in enumerate(matches) if x != "x"]
        if len(to_compare) < 2:
            continue
        j, x = to_compare[0]
        first = "%%(lv%(j)s)s_n%(x)s" % locals()
        cond = " || ".join(["%(first)s != %%(lv%(j)s)s_n%(x)s" % locals() for j, x in to_compare[1:]])
        if cond:
            check += """
            if (%(cond)s) {
                PyErr_SetString(PyExc_ValueError, "Input dimensions do not match.");
                %%(fail)s
            }
            """ % locals()
    return init % sub + check % sub


def make_alloc(loop_orders, dtype, sub):
    nd = len(loop_orders[0])
    init_dims = ""
    for i, candidates in enumerate(zip(*loop_orders)):
        for j, candidate in enumerate(candidates):
            if candidate != 'x':
                var = sub['lv%i' % j]
                init_dims += "dims[%(i)s] = %(var)s_n%(candidate)s;\n" % locals()
                break
        else:
            init_dims += "dims[%(i)s] = 1;\n" % locals()
            #raise Exception("For each looping dimension, at least one input must have a non-broadcastable dimension.")
    return """
    {
        npy_intp dims[%(nd)s];
        //npy_intp* dims = (npy_intp*)malloc(%(nd)s * sizeof(npy_intp));
        %(init_dims)s
        if (!%(olv)s) {
            %(olv)s = (PyArrayObject*)PyArray_EMPTY(%(nd)s, dims, type_num_%(olv)s, 0);
        }
        else {
            PyArray_Dims new_dims;
            new_dims.len = %(nd)s;
            new_dims.ptr = dims;
            PyObject* success = PyArray_Resize(%(olv)s, &new_dims, 0, PyArray_CORDER);
            if (!success) {
                // If we can't resize the ndarray we have we can allocate a new one.
                PyErr_Clear();
                Py_XDECREF(%(olv)s);
                %(olv)s = (PyArrayObject*)PyArray_EMPTY(%(nd)s, dims, type_num_%(olv)s, 0);
            }
        }
        if (!%(olv)s) {
            %(fail)s
        }
    }
    """ % dict(locals(), **sub)


def make_loop(loop_orders, dtypes, loop_tasks, sub):
    """
    Make a nested loop over several arrays and associate specific code
    to each level of nesting.
    
    @type loop_orders: list of N tuples of length M.
    @param loop_orders: Each value of each
      tuple can be either the index of a dimension to loop over or
      the letter 'x' which means there is no looping to be done
      over that variable at that point (in other words we broadcast
      over that dimension). If an entry is an integer, it will become
      an alias of the entry of that rank.

    @type loop_tasks: list of M+1 pieces of code.
    @param loop_tasks: The ith loop_task is code
      to be executed just before going to the next element of the
      ith dimension. The last is code to be executed at the very end.

    @type sub: a dictionary.
    @param sub: Maps 'lv#' to a suitable variable name.
      The 'lvi' variable corresponds to the ith element of loop_orders.
    """

    def loop_over(preloop, code, indices, i):
        iterv = 'ITER_%i' % i
        update = ""
        suitable_n = "1"
        for j, index in enumerate(indices):
            var = sub['lv%i' % j]
            update += "%(var)s_iter += %(var)s_jump%(index)s_%(i)s;\n" % locals()
            if index != 'x':
                suitable_n = "%(var)s_n%(index)s" % locals()
        return """
        %(preloop)s
        for (int %(iterv)s = %(suitable_n)s; %(iterv)s; %(iterv)s--) {
            %(code)s
            %(update)s
        }
        """ % locals()

    preloops = {}
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes)):
        for j, index in enumerate(loop_order):
            if index != 'x':
                preloops.setdefault(j, "")
                preloops[j] += ("%%(lv%(i)s)s_iter = (%(dtype)s*)(%%(lv%(i)s)s->data);\n" % locals()) % sub
                break
        else: # all broadcastable
            preloops.setdefault(0, "")
            preloops[0] += ("%%(lv%(i)s)s_iter = (%(dtype)s*)(%%(lv%(i)s)s->data);\n" % locals()) % sub

    if len(loop_tasks) == 1:
        s = preloops.get(0, "")
    else:
        s = ""
        for i, (pre_task, task), indices in reversed(zip(xrange(len(loop_tasks) - 1), loop_tasks, zip(*loop_orders))):
            s = loop_over(preloops.get(i, "") + pre_task, s + task, indices, i)
    
    s += loop_tasks[-1]
    return "{%s}" % s



# print make_declare(((0, 1, 2, 3), ('x', 1, 0, 3), ('x', 'x', 'x', 0)),
#                    ('double', 'int', 'float'),
#                    dict(lv0='x', lv1='y', lv2='z', fail="FAIL;"))

# print make_checks(((0, 1, 2, 3), ('x', 1, 0, 3), ('x', 'x', 'x', 0)),
#                   ('double', 'int', 'float'),
#                   dict(lv0='x', lv1='y', lv2='z', fail="FAIL;"))

# print make_alloc(((0, 1, 2, 3), ('x', 1, 0, 3), ('x', 'x', 'x', 0)),
#                  'double',
#                  dict(olv='out', lv0='x', lv1='y', lv2='z', fail="FAIL;"))

# print make_loop(((0, 1, 2, 3), ('x', 1, 0, 3), ('x', 'x', 'x', 0)),
#                 ('double', 'int', 'float'),
#                 (("C00;", "C%01;"), ("C10;", "C11;"), ("C20;", "C21;"), ("C30;", "C31;"),"C4;"),
#                 dict(lv0='x', lv1='y', lv2='z', fail="FAIL;"))

# print make_loop(((0, 1, 2, 3), (3, 'x', 0, 'x'), (0, 'x', 'x', 'x')),
#                 ('double', 'int', 'float'),
#                 (("C00;", "C01;"), ("C10;", "C11;"), ("C20;", "C21;"), ("C30;", "C31;"),"C4;"),
#                 dict(lv0='x', lv1='y', lv2='z', fail="FAIL;"))


##################
### DimShuffle ###
##################



#################
### Broadcast ###
#################




################
### CAReduce ###
################






