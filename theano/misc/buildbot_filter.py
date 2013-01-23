#!/usr/bin/env python
import sys


def filter_output(fd_in):
    s = ""
    for line in fd_in:
        toks = line.split()
        if len(toks):
            if toks[0] == "File" and toks[-1].startswith('test'):
                s += line
            elif toks[0].startswith("ImportError"):
                s += line
            elif toks[0] in ["KnownFailureTest:", "Exception:", "Failure:",
                           "AssertionError", "AssertionError:"]:
                s += line
            elif toks[0] == "Executing" and toks[1] in ["tests", 'nosetests']:
                s += line
    return s

if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    if len(sys.argv) > 1:
        print filter_output(open(sys.argv[1]))
    else:
        print filter_output(sys.stdin)
