import sys

if sys.version_info[:2] >= (2,5):
    from collections import deque
else:
    class deque(object):
        def __init__(self, iterable=(), maxsize=-1):
            if not hasattr(self, 'data'):
                self.left = self.right = 0
                self.data = {}
            self.maxsize = maxsize
            self.extend(iterable)
        
        def append(self, x):
            self.data[self.right] = x
            self.right += 1
            if self.maxsize != -1 and len(self) > self.maxsize:
                self.popleft()
                
        def remove(self, x):
            if self.left == self.right:
                raise ValueError('cannot remove from empty deque')
            for i in xrange(self.left, self.right-1):
                elem = self.data[i]
                if elem==x:
                    self.__delitem__(i)
                    break
                
        def appendleft(self, x):
            self.left -= 1        
            self.data[self.left] = x
            if self.maxsize != -1 and len(self) > self.maxsize:
                self.pop()      
        
        def pop(self):
            if self.left == self.right:
                raise IndexError('cannot pop from empty deque')
            self.right -= 1
            elem = self.data[self.right]
            del self.data[self.right]         
            return elem
    
        def popleft(self):
            if self.left == self.right:
                raise IndexError('cannot pop from empty deque')
            elem = self.data[self.left]
            del self.data[self.left]
            self.left += 1
            return elem

        def clear(self):
            self.data.clear()
            self.left = self.right = 0

        def extend(self, iterable):
            for elem in iterable:
                self.append(elem)

        def extendleft(self, iterable):
            for elem in iterable:
                self.appendleft(elem)

        def rotate(self, n=1):
            if self:
                n %= len(self)
                for i in xrange(n):
                    self.appendleft(self.pop())

        def __getitem__(self, i):
            if i < 0:
                i += len(self)
            try:
                return self.data[i + self.left]
            except KeyError:
                raise IndexError

        def __setitem__(self, i, value):
            if i < 0:
                i += len(self)        
            try:
                self.data[i + self.left] = value
            except KeyError:
                raise IndexError

        def __delitem__(self, i):
            size = len(self)
            data = self.data
            if i < 0:
                i += size
            if not data.has_key(i):
                raise IndexError
            for j in xrange(self.left+i, self.right-1):
                data[j] = data[j+1]
            self.pop()
    
        def __len__(self):
            return self.right - self.left

        def __cmp__(self, other):
            if type(self) != type(other):
                return cmp(type(self), type(other))
            return cmp(list(self), list(other))
            
        def __repr__(self, _track=[]):
            if id(self) in _track:
                return '...'
            _track.append(id(self))
            r = 'deque(%r)' % (list(self),)
            _track.remove(id(self))
            return r
    
        def __getstate__(self):
            return (tuple(self),)
    
        def __setstate__(self, s):
            self.__init__(s[0])
        
        def __hash__(self):
            raise TypeError
    
        def __copy__(self):
            return self.__class__(self)
    
        def __deepcopy__(self, memo={}):
            from copy import deepcopy
            result = self.__class__()
            memo[id(self)] = result
            result.__init__(deepcopy(tuple(self), memo))
            return result

from cc import \
    CLinker, OpWiseCLinker, DualLinker

import compiledir # adds config vars

from env import \
    InconsistencyError, Env

from destroyhandler import \
    DestroyHandler 

from graph import \
    Apply, Variable, Constant, Value, view_roots

from link import \
    Container, Linker, LocalLinker, PerformLinker, WrapLinker, WrapLinkerMany

from op import \
    Op

from opt import (Optimizer, optimizer, SeqOptimizer,
    MergeOptimizer, MergeOptMerge, 
    LocalOptimizer, local_optimizer, LocalOptGroup, 
    OpSub, OpRemove, PatternSub, 
    NavigatorOptimizer, TopoOptimizer, EquilibriumOptimizer, 
    InplaceOptimizer, PureThenInplaceOptimizer, 
    OpKeyOptimizer)

from optdb import \
    DB, Query, \
    EquilibriumDB, SequenceDB

from toolbox import \
    Bookkeeper, History, Validator, ReplaceValidate, NodeFinder, PrintListener

from type import \
    Type, Generic, generic

from utils import \
    object2, MethodNotDefined

