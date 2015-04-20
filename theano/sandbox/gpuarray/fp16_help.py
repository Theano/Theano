from theano import scalar


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


class Cast16(scalar.Cast):
    def c_code(self, node, name, inputs, outputs, sub):
        return "%s = %s;\n" % (outputs[0], inputs[0])
