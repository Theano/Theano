
from gen_oplist import print_title, print_hline


def print_file(file):

    print >>file, '.. _typelist:\n\n'

    print_title(file, "Type List", "~", "~")

    print >>file, "*THIS PAGE IS A PLACEHOLDER: WRITEME*"
    print >>file, ""
    print_hline(file)

    print >>file, ""
    print >>file, ".. contents::"
    print >>file, ""

    print_title(file, "Type Classes", '=')

    print >>file, "- scalar.Scalar\n"
    print >>file, "- tensor.Tensor\n"
    print >>file, "- sparse.Sparse\n"

    print_title(file, "Type Instances", '=')

    print >>file, "- scalar.int8\n"
    print >>file, "- tensor.lvector\n"
    print >>file, "- sparse.??\n"

    print >>file, ""


if __name__ == '__main__':

    if len(sys.argv) >= 2:
        file = open(sys.argv[1], 'w')
    else:
        file = sys.stdout

    print_file(file)
