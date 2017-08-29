from __future__ import absolute_import, print_function, division
import os
import sys


class PathParser(object):
    """
    Class that allows to modify system's PATH environment variable
    at runtime. Currently used in ``theano.gpuarray.dnn`` module
    on Windows only.

    **Examples**:

    ..code-block:: python

        theano.pathparse.PathParser(pathToAdd1, pathToAdd2, ...)
        # PATH is then automatically updated for this execution.


    ..code-block:: python

        paths = theano.pathparse.PathParser()
        paths.add(path1)
        paths.add(path2)
        # PATH is updated after each call to ``add()``.

    """
    paths = set()

    def _add(self, path):
        path = path.strip()
        if path:
            if sys.platform == 'win32':
                # Windows is case-insensitive.
                path = path.lower()
            self.paths.add(os.path.abspath(path))

    def _update(self):
        os.environ['PATH'] = os.pathsep.join(sorted(self.paths))

    def _parse(self):
        for path in os.environ['PATH'].split(os.pathsep):
            self._add(path)

    def __init__(self, *paths):
        self._parse()
        for path in paths:
            self._add(path)
        self._update()

    def add(self, path):
        self._add(path)
        self._update()

    def _debug(self):
        for path in sorted(self.paths):
            print(path)
