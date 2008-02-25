import sys
if sys.version_info[:2] < (2,5):
    def all(iterable):
        for element in iterable:
            if not element:
                return False
        return True
else:
     # Only bother with this else clause and the __all__ line if you are putting
     # this in a separate file.
     import __builtin__
     all = __builtin__.all

__all__ = ['all']
