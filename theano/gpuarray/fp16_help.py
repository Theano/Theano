from __future__ import absolute_import, print_function, division


def work_dtype(dtype):
    if dtype == 'float16':
        return 'float32'
    else:
        return dtype


def load_w(dtype):
    if dtype == 'float16':
        return '__half2float'
    else:
        return ''


def write_w(dtype):
    if dtype == 'float16':
        return '__float2half_rn'
    else:
        return ''
