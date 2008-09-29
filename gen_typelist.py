
from gen_oplist import print_title, print_hline


if __name__ == '__main__':
    print_title("Type List", "~", "~")

    print "*THIS PAGE IS A PLACEHOLDER: WRITEME*"
    print ""
    print_hline()

    print ""
    print ".. contents::"
    print ""

    print_title("Type Classes", '=')

    print "- scalar.Scalar\n"
    print "- tensor.Tensor\n"
    print "- sparse.Sparse\n"

    print_title("Type Instances", '=')

    print "- scalar.int8\n"
    print "- tensor.lvector\n"
    print "- sparse.??\n"

    print ""

    for line in open("doc/header.txt"):
        print line[:-1]
