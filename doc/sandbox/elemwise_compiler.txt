.. _sandbox_elemwise:

=================
Elemwise compiler
=================

'''Stale specification page.  Upgrade this to provide useful developer doc. 2008.09.04'''
== Definitions ==

The elementwise compiler takes inputs {{{(in0, in1, in2, ...)}}}, outputs {{{(out0, out1, out2, ...)}}}, broadcast modes {{{(mod0, mod1, mod2, ...)}}} where each mode corresponds to an output as well as {{{order}}} which determines if we broadcast/accumulate over the first or last dimensions (the looping order, basically, but some operations are only valid for one particular order!).

The broadcast mode serves to calculate the rank of the corresponding output and how to map each input element to an output element:

  * {{{broadcast}}}
    * output.rank = max(input.rank)
    * the inputs of lesser rank are broadcasted over missing dimensions
    * if {{{order == f}}} ([3, 5], [5]) => [3, 5] or ([7, 8, 9], [8, 9]) => [7, 8, 9]
    * if {{{order == c}}} ([3, 5], [3]) => [3, 5] or ([7, 8, 9], [7, 8]) => [7, 8, 9]
  * {{{(accumulate, Accumulator)}}}
    * output.rank = min(input.rank)
    * for the inputs of greater rank, we use Accumulator (sum, product, etc.) to accumulate over the first dimensions

      * e.g. {{{if Accumulator == sum, order == c, x.rank == 2, y.rank == 1 and z = f(x, y) then z[i] = f(sum_j(x[i, j]), y[i])}}}

    * if {{{order == f}}} ([3, 5], [5]) => [5] or ([7, 8, 9], [8, 9]) => [8, 9]
    * if {{{order == c}}} ([3, 5], [3]) => [3] or ([7, 8, 9], [7, 8]) => [7, 8]

{{{order == c}}} is equivalent to transposing the outputs of an {{{order == f}}} operation on transposed inputs.

This does not cover all cases of broadcasting, but I believe they cover enough. Other cases of broadcasting can be emulated with proper transposition and/or slicing.
 * Could you give some examples of what kinds of broadcasting are and are not covered by your proposed implementation?

  * For rank <= 2, I think only operations of the form {{{add(ones(3,1), ones(1,3)))}}} are missing. I actually didn't think of that one before now.
  * In general, it only handles f(shape(head, ...), shape(head, ...), ...) and f(shape(..., tail), shape(..., tail), ...)
  * Maybe I could add a general case later... the thing is that I think the ones I am considering here are easier to streamline.

Point of clarification: the order discussed here corresponds to a set of broadcasting rules, and is independent from the storage order.  The 'f' order corresponds to numpy's broadcasting rules, while the 'c' order is something new and different (TODO VERIFY!)

Question: does it make sense to apply the order to the loop, or is this broadcast order something which will be local to each input argument.  What happens when the elemwise compiler deals with more complex subgraphs with multiple inputs and outputs?

== The loop ==

Here is the loop for {{{order == c}}}. Check for errors!

.. code-block:: cpp

    <initialize iterators>

    i1 = -1
    while (++i1 < dim1) {
      i2 = -1
      rank_N-1_accumulator = init
      while (++i2 < dim2) {
        ...
        iN = -1
        while (++iN < dimN) {
          <accumulate rank N input>
          <SET rank N output using broadcasted inputs>
          <NEXT rank N iterator>
        }
        ...
      }
      <SET rank 1 output using accumulated inputs>
      <NEXT rank 1 iterator>
    }

When {{{order == f}}}, the iterators ''ideally'' (but not necessarily) iterate in FORTRAN order, i.e. the while loops are on {{{dimN..dim1}}} instead of {{{dim1..dimN}}}.

{{{order}}} does __not__ represent the {{{C/F_CONTIGUOUS}}} flags of the inputs or outputs. Depending on combinations of those parameters, different loops will be used. If {{{order == f and C_CONTIGUOUS(array)}}}, for example, the loop will be on {{{dim1..dimN}}} and the matrices of lesser rank will need to be looped over several times.

An Optimizer should look at the operations in the graph and figure out whether to allocate C_CONTIGUOUS (ideal for {{{order == c}}}) or F_CONTIGUOUS (ideal for {{{order == f}}}) arrays.

== Gradient ==

The input ranks become the output ranks and gradients of the same rank as the outputs are added to the input list. If an output was given mode {{{broadcast}}}, then all inputs used to calculate it had to be broadcasted to that shape, so we must sum over the broadcasted dimensions on the gradient. The mode that we give to those inputs is therefore {{{(accumulate, sum)}}}. Inversely, if an output was given mode {{{(accumulate, sum)}}}, then all inputs used to calculate it had to be summed over those dimensions. Therefore, we give them mode {{{broadcast}}} in grad. Other accumulators than sum might prove more difficult. For example, the ith gradient for product is grad*product/x_i. Not sure how to handle that automatically.
 * I don't exactly follow this paragraph, but I think I catch the general idea and it seems to me like it will work very well.

  * In a nutshell for {{{broadcast}}} I calculate the gradient as normal assuming the shape is broadcasted and then I sum over what I had to broadcast.

 * Could you explain why the accumulator gradient (e.g. product) can be trickier?

  * I thought about it and I figured that the general case is {{{g_accum[N-i+1], g_m[i] = grad_fn(accum[i-1], m[i], g_accum[N-i])}}} where {{{g_accum}}} is the accumulated gradient wrt the accumulator {{{accum}}}. It can be short-circuited in sum and product's case: for sum, grad_fn is the identity on its last argument so {{{g_m[i] == g_accum[i] == g_accum[0] == g_z for all i}}}. In product's case, {{{accum[i-1] == product(m[1:i-1]) and g_accum[N-i] == g_z * product(m[i+1:N])}}}, multiply them together and you obtain {{{g_z * product(m)/m[i]}}} where obviously we only need to compute {{{product(m)}}} once. It's worth handling those two special cases, for the general case I don't know.


