#section support_code_struct

int APPLY_SPECIFIC(ctc_cost_gpu)(PyGpuArrayObject   *  in_activations,
                                 PyGpuArrayObject   *  in_labels,
                                 PyGpuArrayObject   *  in_input_lengths,
                                 PyGpuArrayObject   ** out_costs,
                                 PyGpuArrayObject   ** out_gradients,
                                 PyGpuContextObject *  ctx)
{
   return 0;
}

int APPLY_SPECIFIC(ctc_cost_gpu_no_grad)(PyGpuArrayObject   *  in_activations,
                                         PyGpuArrayObject   *  in_labels,
                                         PyGpuArrayObject   *  in_input_lengths,
                                         PyGpuArrayObject   ** out_costs,
                                         PyGpuContextObject *  ctx)
{
    return APPLY_SPECIFIC(ctc_cost_gpu)(in_activations,
                                        in_labels,
                                        in_input_lengths,
                                        out_costs,
                                        NULL,
                                        ctx);
}