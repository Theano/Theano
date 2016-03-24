#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
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
                             "AssertionError", "AssertionError:",
                             "GradientError:"]:
                s += line
            elif toks[0] == "Executing" and toks[1] in ["tests", 'nosetests']:
                s += line
    return s

if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            print(filter_output(f))
    else:
        print(filter_output(sys.stdin))
