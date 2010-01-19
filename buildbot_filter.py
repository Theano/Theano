#!/usr/bin/env python
import sys
for line in sys.stdin:
    toks = line.split()
    if len(toks):
        if toks[0] == "File" and toks[-1].startswith('test'):
            print line,
        if toks[0].startswith("ImportError"):
            print line,

