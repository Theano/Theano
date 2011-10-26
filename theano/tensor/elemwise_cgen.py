

def make_declare(loop_orders, dtypes, sub):
    """
    Produce code to declare all necessary variables.
    """

    decl = ""
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes)):
        var = sub['lv%i' % i] # input name corresponding to ith loop variable
        # we declare an iteration variable
        # and an integer for the number of dimensions
        decl += """
        %(dtype)s* %(var)s_iter;
        int %(var)s_nd;
        """ % locals()
        for j, value in enumerate(loop_order):
            if value != 'x':
                # If the dimension is not broadcasted, we declare
                # the number of elements in that dimension,
                # the stride in that dimension,
                # and the jump from an iteration to the next
                decl += """
                int %(var)s_n%(value)i;
                int %(var)s_stride%(value)i;
                int %(var)s_jump%(value)i_%(j)i;
                """ % locals()
            else:
                # if the dimension is broadcasted, we only need
                # the jump (arbitrary length and stride = 0)
                decl += """
                int %(var)s_jump%(value)s_%(j)i;
                """ % locals()

    return decl


def make_checks(loop_orders, dtypes, sub):
    init = ""
    for i, (loop_order, dtype) in enumerate(zip(loop_orders, dtypes)):
        var = "%%(lv%i)s" % i
        # List of dimensions of var that are not broadcasted
        nonx = [x for x in loop_order if x != 'x']
        if nonx:
            # If there are dimensions that are not broadcasted
            # this is a check that the number of dimensions of the
            # tensor is as expected.
            min_nd = max(nonx) + 1
            init += """
            if (%(var)s->nd < %(min_nd)s) {
                PyErr_SetString(PyExc_ValueError, "Not enough dimensions on input.");
                %%(fail)s
            }
            """ % locals()

        # In loop j, adjust represents the difference of values of the
        # data pointer between the beginning and the end of the
        # execution of loop j+1 (the loop inside the current one). It
        # is equal to the stride in loop j+1 times the length of loop
        # j+1, or 0 for the inner-most loop.
        adjust = "0"

        # We go from the inner loop to the outer loop
        for j, index in reversed(list(enumerate(loop_order))):
            if index != 'x':
                # Initialize the variables associated to the jth loop
                # jump = stride - adjust
                jump = "(%s) - (%s)" % ("%(var)s_stride%(index)s" % locals(), adjust)
                init += """
                %(var)s_n%(index)s = %(var)s->dimensions[%(index)s];
                %(var)s_stride%(index)s = %(var)s->strides[%(index)s] / sizeof(%(dtype)s);
                %(var)s_jump%(index)s_%(j)s = %(jump)s;
                //printf("%(var)s_jump%(index)s_%(j)s is:");
                //std::cout << %(var)s_jump%(index)s_%(j)s << std::endl;
                """ % locals()
                adjust = "%(var)s_n%(index)s*%(var)s_stride%(index)s" % locals()
            else:
                jump = "-(%s)" % adjust
                init += """
                %(var)s_jump%(index)s_%(j)s = %(jump)s;
                //printf("%(var)s_jump%(index)s_%(j)s is:");
                //std::cout << %(var)s_jump%(index)s_%(j)s << std::endl;
                """ % locals()
                adjust = "0"
    check = ""

    # This loop builds multiple if conditions to verify that the
    # dimensions of the inputs match, and the first one that is true
    # raises an informative error message
    for matches in zip(*loop_orders):
        to_compare = [(j, x) for j, x in enumerate(matches) if x != "x"]

        #elements of to_compare are pairs ( input_variable_idx, input_variable_dim_idx )
        if len(to_compare) < 2:
            continue
        j0, x0 = to_compare[0]
        for (j, x) in to_compare[1:]:
            check += """
            if (%%(lv%(j0)s)s_n%(x0)s != %%(lv%(j)s)s_n%(x)s)
            {
                PyErr_Format(PyExc_ValueError, "Input dimension mis-match. (input[%%%%i].shape[%%%%i] = %%%%i, input[%%%%i].shape[%%%%i] = %%%%i)",
                   %(j0)s,
                   %(x0)s,
                   %%(lv%(j0)s)s_n%(x0)s,
                   %(j)s,
                   %(x)s,
                   %%(lv%(j)s)s_n%(x)s
                );
                %%(fail)s
            }
            """ % locals()

    return init % sub + check % sub


def make_alloc(loop_orders, dtype, sub):
    """
    Generate C code to allocate outputs.
    """

    nd = len(loop_orders[0])
    init_dims = ""
    # For each dimension, the tensors are either all broadcasted, in
    # which case the output will also be broadcastable (dimension =
    # 1), or one or more are not broadcasted, in which case the number
    # of elements of the output in that dimension will be equal to the
    # number of elements of any of them.
    for i, candidates in enumerate(zip(*loop_orders)):
        for j, candidate in enumerate(candidates):
            if candidate != 'x':
                var = sub['lv%i' % j]
                init_dims += "dims[%(i)s] = %(var)s_n%(candidate)s;\n" % locals()
                break
        else:
            init_dims += "dims[%(i)s] = 1;\n" % locals()
            #raise Exception("For each looping dimension, at least one input must have a non-broadcastable dimension.")

    # TODO: it would be interesting to allocate the output in such a
    # way that its contiguous dimensions match one of the input's
    # contiguous dimensions, or the dimension with the smallest
    # stride. Right now, it is allocated to be C_CONTIGUOUS.

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
    @param loop_tasks: The ith loop_task is a pair of strings, the first
      string is code to be executed before the ith loop starts, the second
      one contains code to be executed just before going to the next element
      of the ith dimension.
      The last element if loop_tasks is a single string, containing code
      to be executed at the very end.

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


def make_reordered_loop(init_loop_orders, olv_index, dtypes, inner_task, sub):
    '''A bit like make_loop, but when only the inner-most loop executes code.

    All the loops will be reordered so that the loops over the output tensor
    are executed with memory access as contiguous as possible.
    For instance, if the output tensor is c_contiguous, the inner-most loop
    will be on its rows; if it's f_contiguous, it will be on its columns.

    The output tensor's index among the loop variables is indicated by olv_index.
    '''

    # Number of variables
    nvars = len(init_loop_orders)
    # Number of loops (dimensionality of the variables)
    nnested = len(init_loop_orders[0])

    # This is the var from which we'll get the loop order
    ovar = sub['lv%i' % olv_index]

    # The loops are ordered by (decreasing) absolute values of ovar's strides.
    # The first element of each pair is the absolute value of the stride
    # The second element correspond to the index in the initial loop order
    order_loops = """
    std::vector< std::pair<int, int> > %(ovar)s_loops(%(nnested)i);
    std::vector< std::pair<int, int> >::iterator %(ovar)s_loops_it = %(ovar)s_loops.begin();
    """ % locals()

    # Fill the loop vector with the appropriate <stride, index> pairs
    for i, index in enumerate(init_loop_orders[olv_index]):
        if index != 'x':
            order_loops += """
            %(ovar)s_loops_it->first = abs(%(ovar)s->strides[%(index)i]);
            """  % locals()
        else:
            # Stride is 0 when dimension is broadcastable
            order_loops += """
            %(ovar)s_loops_it->first = 0;
            """ % locals()

        order_loops += """
        %(ovar)s_loops_it->second = %(i)i;
        ++%(ovar)s_loops_it;
        """ % locals()

    # We sort in decreasing order so that the outermost loop (loop 0)
    # has the largest stride, and the innermost loop (nnested - 1) has
    # the smallest stride.
    order_loops += """
    // rbegin and rend are reversed iterators, so this sorts in decreasing order
    std::sort(%(ovar)s_loops.rbegin(), %(ovar)s_loops.rend());
    """ % locals()

    ## Get the (sorted) total number of iterations of each loop
    # Get totals in the initial order
    # For each dimension, the tensors are either all broadcasted, in
    # which case there is only one iteration of the loop, or one or
    # more are not broadcasted, in which case the number of elements
    # of any of them will be equal to the number of iterations we have
    # to do.
    totals = []
    for i, candidates in enumerate(zip(*init_loop_orders)):
        for j, candidate in enumerate(candidates):
            if candidate != 'x':
                var = sub['lv%i' % j]
                total = "%(var)s_n%(candidate)s" % locals()
                break
        else:
            total = '1';
        totals.append(total)

    declare_totals = """
    int init_totals[%(nnested)s] = {%(totals)s};
    """ % dict(
            nnested = nnested,
            totals = ', '.join(totals)
            )

    # Sort totals to match the new order that was computed by sorting
    # the loop vector. One integer variable per loop is declared.
    declare_totals += """
    %(ovar)s_loops_it = %(ovar)s_loops.begin();
    """ % locals()

    for i in xrange(nnested):
        declare_totals += """
        int TOTAL_%(i)i = init_totals[%(ovar)s_loops_it->second];
        ++%(ovar)s_loops_it;
        """ % locals()

    ## Get sorted strides and jumps
    # Get strides in the initial order
    def get_loop_strides(loop_order, i):
        """
        Returns a list containing a C expression representing the
        stride for each dimension of the ith variable, in the
        specified loop_order.
        """
        var = sub["lv%i" % i]
        r = []
        for index in loop_order:
            # Note: the stride variable is not declared for broadcasted variables
            if index != 'x':
                r.append("%(var)s_stride%(index)s" % locals())
            else:
                r.append('0')
        return r

    # We declare the initial strides as a 2D array, nvars x nnested
    declare_strides_jumps = """
    int init_strides[%(nvars)i][%(nnested)i] = {
        %(strides)s
    };""" % dict(
            nvars = nvars,
            nnested = nnested,
            strides = ', \n'.join(
                ', '.join(get_loop_strides(lo, i))
                for i, lo in enumerate(init_loop_orders)
                if len(lo)>0))

    # Declare (sorted) stride and jumps for each variable
    # we iterate from innermost loop to outermost loop
    declare_strides_jumps += """
    std::vector< std::pair<int, int> >::reverse_iterator %(ovar)s_loops_rit;
    """ % locals()

    for i in xrange(nvars):
        var = sub["lv%i" % i]
        declare_strides_jumps += """
        %(ovar)s_loops_rit = %(ovar)s_loops.rbegin();""" % locals()

        adjust = "0"
        for j in reversed(range(nnested)):
            jump = "(%s) - (%s)" % ("%(var)s_stride_l%(j)i" % locals(), adjust)
            declare_strides_jumps +="""
            int %(var)s_stride_l%(j)i = init_strides[%(i)i][%(ovar)s_loops_rit->second];
            int %(var)s_jump_l%(j)i = %(jump)s;
            ++%(ovar)s_loops_rit;
            """ % locals()
            adjust = "TOTAL_%(j)i * %(var)s_stride_l%(j)i" % locals()

    declare_iter = ""
    for i, dtype in enumerate(dtypes):
        var = sub["lv%i" % i]
        declare_iter += "%(var)s_iter = (%(dtype)s*)(%(var)s->data);\n" % locals()

    loop = inner_task
    for i in reversed(range(nnested)):
        iterv = 'ITER_%i' % i
        total = 'TOTAL_%i' % i
        update = ''
        for j in xrange(nvars):
            var = sub["lv%i" % j]
            update += "%(var)s_iter += %(var)s_jump_l%(i)i;\n" % locals()

        loop = """
        for (int %(iterv)s = %(total)s; %(iterv)s; %(iterv)s--)
        { // begin loop %(i)i
            %(loop)s
            %(update)s
        } // end loop %(i)i
        """ % locals()

    return '\n'.join([
            '{',
            order_loops,
            declare_totals,
            declare_strides_jumps,
            declare_iter,
            loop,
            '}\n',
            ])

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






