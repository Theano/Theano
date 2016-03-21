import theano
import theano.tensor as T

def scan_with_checkpoints(fn, sequences=[], outputs_info=None,
                          non_sequences=[], name="checkpointscan_fn",
                          n_steps=None, save_every_N=10):

    """
    Current assumptions : 
    - Every sequence has the same length
    - If n_steps is specified, it has the same value as the length of any sequence
    - The value of "save_every_N" divides the number of steps the Scan will
      run without remainder
    - Only singly-recurrent and non-recurrent outputs are used.
      No multiple recurrences.
    - Only the last timestep of any output will ever be used.
    """
    
    # Standardize the format of input arguments
    if not isinstance(sequences, list):
        sequences = [sequences]
    if not isinstance(outputs_info, list):
        outputs_info = [outputs_info]
    if not isinstance(non_sequences, list):
        non_sequences = [non_sequences]
    
    # Determine how many steps the original scan would run
    if n_steps is None:
        n_steps = sequences[0].shape[0]
    else:
        n_steps = n_steps

    # Compute the number of steps of the inner and of the outer scan
    o_n_steps = n_steps / save_every_N
    i_n_steps = save_every_N

    # Establish the input variables of the outer scan
    o_sequences = [s.reshape([s.shape[0] / save_every_N, save_every_N] +
                             [s.shape[i] for i in range(1, s.ndim)], s.ndim + 1) for s in sequences]
    new_nitsots = [i for i in outputs_info if i is None]
    new_sitsots = [i for i in outputs_info if i is not None]
    o_nonsequences = non_sequences + [i_n_steps]

    def outer_step(*args):
        # Separate the received arguments into their respective (seq, outputs
        # from previous iterations, nonseqs) categories
        i_sequences = list(args[:len(o_sequences)])
        i_prev_outputs = list(args[len(o_sequences):-len(o_nonsequences)])
        i_non_sequences = list(args[-len(o_nonsequences):])
        
        # Assemble the correct outputs_info list for the inner_scan
        i_outputs_info = []

        # Call the user-provided function with the proper arguments
        results, updates = theano.scan(fn=fn,
                                       sequences=i_sequences,
                                       outputs_info=i_prev_outputs + [None,] * len(new_nitsots),
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
                                  
    # Keep only the last timestep of every output but keep all the updates
    return results, updates
    if not isinstance(results, list):
        return results[-1:], updates
    else:
        return [r[-1:] for r in results], updates
