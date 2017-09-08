from __future__ import absolute_import, print_function, division


def work_dtype(dtype):
    """
    Return the data type for working memory.

    """
    if dtype == 'float16':
        return 'float32'
    else:
        return dtype


def load_w(dtype):
    """
    Return the function name to load data.

    This should be used like this::

        code = '%s(ival)' % (load_w(input_type),)

    """
    if dtype == 'float16':
        return 'ga_half2float'
    else:
        return ''


def write_w(dtype):
    """
    Return the function name to write data.

    This should be used like this::

        code = 'res = %s(oval)' % (write_w(output_type),)

    """
    if dtype == 'float16':
        return 'ga_float2half'
    else:
        return ''
