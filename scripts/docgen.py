
import sys
import os

throot = "/".join(sys.path[0].split("/")[:-1])

os.chdir(throot)

def mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

mkdir("html")
mkdir("html/doc")
mkdir("html/api")

os.system("epydoc --config doc/api/epydoc.conf -o html/api")

import sphinx
sys.path[0:0] = [os.path.realpath('doc')]
sphinx.main([sys.argv[0], 'doc', 'html'])


