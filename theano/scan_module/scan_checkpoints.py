from __future__ import absolute_import, print_function, division

import theano


def scan_checkpoints(fn, sequences=[], outputs_info=None, non_sequences=[],
                     name="checkpointscan_fn", n_steps=None, save_every_N=10):
    """Scan function that uses less memory, but is more restrictive.

    In :func:`~theano.scan`, if you compute the gradient of the output
    with respect to the input, you will have to store the intermediate
    results at each time step, which can be prohibitively huge. This
    function allows to do ``save_every_N`` steps of forward computations
    without storing the intermediate results, and to recompute them during
    the gradient computation.

    Notes
    -----
    Current assumptions:

    * Every sequence has the same length.
    * If ``n_steps`` is specified, it has the same value as the length of
      any sequence.
    * The value of ``save_every_N`` divides the number of steps the scan
      will run without remainder.
    * Only singly-recurrent and non-recurrent outputs are used.
      No multiple recurrences.
    * Only the last timestep of any output will ever be used.

    Parameters
    ----------
    fn
        ``fn`` is a function that describes the operations involved in one
        step of ``scan``. See the documentation of :func:`~theano.scan`
        for more information.

    sequences
        ``sequences`` is the list of Theano variables or dictionaries
        describing the sequences ``scan`` has to iterate over. All
        sequences must be the same length in this version of ``scan``.

    outputs_info
        ``outputs_info`` is the list of Theano variables or dictionaries
        describing the initial state of the outputs computed
        recurrently.

    non_sequences
        ``non_sequences`` is the list of arguments that are passed to
        ``fn`` at each steps. One can opt to exclude variable
        used in ``fn`` from this list as long as they are part of the
        computational graph, though for clarity we encourage not to do so.

    n_steps
        ``n_steps`` is the number of steps to iterate given as an int
        or Theano scalar. If any of the input sequences do not have
        enough elements, scan will raise an error. If the **value is 0**
        the outputs will have **0 rows**. If the value is negative,
        ``scan`` will run backwards in time. If the ``go_backwards`` flag
        is already set and also ``n_steps`` is negative, ``scan`` will run
        forward in time. If n_steps is not provided, ``scan`` will figure
        out the amount of steps it should run given its input sequences.

    save_every_N
        ``save_every_N`` is the number of steps to go without storing
        the computations of ``scan`` (ie they will have to be recomputed
        during the gradient computation).

    Returns
    -------
    tuple
        Tuple of the form ``(outputs, updates)`` as in :func:`~theano.scan`, but
        with a small change: It only contain the output at each
        ``save_every_N`` step. The time steps that are not returned by
        this function will be recomputed during the gradient computation
        (if any).

    See Also
    --------
    :func:`~theano.scan`: Looping in Theano.

    """
    # Standardize the format of input arguments
    if not isinstance(sequences, list):
        sequences = [sequences]
    if not isinstance(outputs_info, list):
        outputs_info = [outputs_info]
    if not isinstance(non_sequences, list):
        non_sequences = [non_sequences]

    # Check that outputs_info has no taps:
    for element in outputs_info:
        if isinstance(element, dict) and 'taps' in element:
            raise RuntimeError("scan_checkpoints doesn't work with taps.")

    # Determine how many steps the original scan would run
    if n_steps is None:
        n_steps = sequences[0].shape[0]

    # Compute the number of steps of the inner and of the outer scan
    o_n_steps = theano.tensor.cast(n_steps / save_every_N, 'int64')
    i_n_steps = save_every_N

    # Establish the input variables of the outer scan
    o_sequences = [s.reshape([s.shape[0] / save_every_N, save_every_N] +
                             [s.shape[i] for i in range(1, s.ndim)],
                             s.ndim + 1) for s in sequences]
    new_nitsots = [i for i in outputs_info if i is None]
    o_nonsequences = non_sequences + [i_n_steps]

    def outer_step(*args):
        # Separate the received arguments into their respective (seq, outputs
        # from previous iterations, nonseqs) categories
        i_sequences = list(args[:len(o_sequences)])
        i_prev_outputs = list(args[len(o_sequences):-len(o_nonsequences)])
        i_non_sequences = list(args[-len(o_nonsequences):])
        i_outputs_infos = i_prev_outputs + [None, ] * len(new_nitsots)

        # Call the user-provided function with the proper arguments
        results, updates = theano.scan(fn=fn,
                                       sequences=i_sequences,
                                       outputs_info=i_outputs_infos,
                                       non_sequences=i_non_sequences[:-1],
                                       name=name + "_inner",
                                       n_steps=i_non_sequences[-1])
        if not isinstance(results, list):
            results = [results]

        # Keep only the last timestep of every output but keep all the updates
        if not isinstance(results, list):
            return results[-1], updates
        else:
            return [r[-1] for r in results], updates

    results, updates = theano.scan(fn=outer_step,
                                   sequences=o_sequences,
                                   outputs_info=outputs_info,
                                   non_sequences=o_nonsequences,
                                   name=name + "_outer",
                                   n_steps=o_n_steps, allow_gc=True)

    return results, updates
