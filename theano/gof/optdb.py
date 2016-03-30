from __future__ import absolute_import, print_function, division
import copy
import sys

from theano.compat import DefaultOrderedDict
from theano.misc.ordered_set import OrderedSet
from six import StringIO, integer_types
from theano.gof import opt
from theano import config


class DB(object):
    def __hash__(self):
        if not hasattr(self, '_optimizer_idx'):
            self._optimizer_idx = opt._optimizer_idx[0]
            opt._optimizer_idx[0] += 1
        return self._optimizer_idx

    def __init__(self):
        self.__db__ = DefaultOrderedDict(OrderedSet)
        self._names = set()
        self.name = None  # will be reset by register
        # (via obj.name by the thing doing the registering)

    def register(self, name, obj, *tags, **kwargs):
        """

        Parameters
        ----------
        name : str
            Name of the optimizer.
        obj
            The optimizer to register.
        tags
            Tag name that allow to select the optimizer.
        kwargs
            If non empty, should contain only use_db_name_as_tag=False.
            By default, all optimizations registered in EquilibriumDB
            are selected when the EquilibriumDB name is used as a
            tag. We do not want this behavior for some optimizer like
            local_remove_all_assert. use_db_name_as_tag=False remove
            that behavior. This mean only the optimizer name and the
            tags specified will enable that optimization.

        """
        # N.B. obj is not an instance of class Optimizer.
        # It is an instance of a DB.In the tests for example,
        # this is not always the case.
        if not isinstance(obj, (DB, opt.Optimizer, opt.LocalOptimizer)):
            raise TypeError('Object cannot be registered in OptDB', obj)
        if name in self.__db__:
            raise ValueError('The name of the object cannot be an existing'
                             ' tag or the name of another existing object.',
                             obj, name)
        if kwargs:
            assert "use_db_name_as_tag" in kwargs
            assert kwargs["use_db_name_as_tag"] is False
        else:
            if self.name is not None:
                tags = tags + (self.name,)
        obj.name = name
        # This restriction is there because in many place we suppose that
        # something in the DB is there only once.
        if obj.name in self.__db__:
            raise ValueError('''You can\'t register the same optimization
multiple time in a DB. Tryed to register "%s" again under the new name "%s".
 Use theano.gof.ProxyDB to work around that''' % (obj.name, name))
        self.__db__[name] = OrderedSet([obj])
        self._names.add(name)
        self.__db__[obj.__class__.__name__].add(obj)
        self.add_tags(name, *tags)

    def add_tags(self, name, *tags):
        obj = self.__db__[name]
        assert len(obj) == 1
        obj = obj.copy().pop()
        for tag in tags:
            if tag in self._names:
                raise ValueError('The tag of the object collides with a name.',
                                 obj, tag)
            self.__db__[tag].add(obj)

    def remove_tags(self, name, *tags):
        obj = self.__db__[name]
        assert len(obj) == 1
        obj = obj.copy().pop()
        for tag in tags:
            if tag in self._names:
                raise ValueError('The tag of the object collides with a name.',
                                 obj, tag)
            self.__db__[tag].remove(obj)

    def __query__(self, q):
        if not isinstance(q, Query):
            raise TypeError('Expected a Query.', q)
        # The ordered set is needed for deterministic optimization.
        variables = OrderedSet()
        for tag in q.include:
            variables.update(self.__db__[tag])
        for tag in q.require:
            variables.intersection_update(self.__db__[tag])
        for tag in q.exclude:
            variables.difference_update(self.__db__[tag])
        remove = OrderedSet()
        add = OrderedSet()
        for obj in variables:
            if isinstance(obj, DB):
                def_sub_query = q
                if q.extra_optimizations:
                    def_sub_query = copy.copy(q)
                    def_sub_query.extra_optimizations = []
                sq = q.subquery.get(obj.name, def_sub_query)

                replacement = obj.query(sq)
                replacement.name = obj.name
                remove.add(obj)
                add.add(replacement)
        variables.difference_update(remove)
        variables.update(add)
        return variables

    def query(self, *tags, **kwtags):
        if len(tags) >= 1 and isinstance(tags[0], Query):
            if len(tags) > 1 or kwtags:
                raise TypeError('If the first argument to query is a Query,'
                                ' there should be no other arguments.',
                                tags, kwtags)
            return self.__query__(tags[0])
        include = [tag[1:] for tag in tags if tag.startswith('+')]
        require = [tag[1:] for tag in tags if tag.startswith('&')]
        exclude = [tag[1:] for tag in tags if tag.startswith('-')]
        if len(include) + len(require) + len(exclude) < len(tags):
            raise ValueError("All tags must start with one of the following"
                             " characters: '+', '&' or '-'", tags)
        return self.__query__(Query(include=include,
                                    require=require,
                                    exclude=exclude,
                                    subquery=kwtags))

    def __getitem__(self, name):
        variables = self.__db__[name]
        if not variables:
            raise KeyError("Nothing registered for '%s'" % name)
        elif len(variables) > 1:
            raise ValueError('More than one match for %s (please use query)' %
                             name)
        for variable in variables:
            return variable

    def __contains__(self, name):
        return name in self.__db__

    def print_summary(self, stream=sys.stdout):
        print("%s (id %i)" % (self.__class__.__name__, id(self)), file=stream)
        print("  names", self._names, file=stream)
        print("  db", self.__db__, file=stream)


class Query(object):
    """

    Parameters
    ----------
    position_cutoff : float
        Used by SequenceDB to keep only optimizer that are positioned before
        the cut_off point.

    """

    def __init__(self, include, require=None, exclude=None,
                 subquery=None, position_cutoff=None,
                 extra_optimizations=None):
        self.include = OrderedSet(include)
        self.require = require or OrderedSet()
        self.exclude = exclude or OrderedSet()
        self.subquery = subquery or {}
        self.position_cutoff = position_cutoff
        if extra_optimizations is None:
            extra_optimizations = []
        self.extra_optimizations = extra_optimizations
        if isinstance(self.require, (list, tuple)):
            self.require = OrderedSet(self.require)
        if isinstance(self.exclude, (list, tuple)):
            self.exclude = OrderedSet(self.exclude)

    def __str__(self):
        return ("Query{inc=%s,ex=%s,require=%s,subquery=%s,"
                "position_cutoff=%d,extra_opts=%s}" %
                (self.include, self.exclude, self.require, self.subquery,
                 self.position_cutoff, self.extra_optimizations))

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'extra_optimizations'):
            self.extra_optimizations = []

    # add all opt with this tag
    def including(self, *tags):
        return Query(self.include.union(tags),
                     self.require,
                     self.exclude,
                     self.subquery,
                     self.position_cutoff,
                     self.extra_optimizations)

    # remove all opt with this tag
    def excluding(self, *tags):
        return Query(self.include,
                     self.require,
                     self.exclude.union(tags),
                     self.subquery,
                     self.position_cutoff,
                     self.extra_optimizations)

    # keep only opt with this tag.
    def requiring(self, *tags):
        return Query(self.include,
                     self.require.union(tags),
                     self.exclude,
                     self.subquery,
                     self.position_cutoff,
                     self.extra_optimizations)

    def register(self, *optimizations):
        return Query(self.include,
                     self.require,
                     self.exclude,
                     self.subquery,
                     self.position_cutoff,
                     self.extra_optimizations + list(optimizations))


class EquilibriumDB(DB):
    """
    A set of potential optimizations which should be applied in an arbitrary
    order until equilibrium is reached.

    Canonicalize, Stabilize, and Specialize are all equilibrium optimizations.

    Parameters
    ----------
    ignore_newtrees
        If False, we will apply local opt on new node introduced during local
        optimization application. This could result in less fgraph iterations,
        but this doesn't mean it will be faster globally.

    Notes
    -----
    We can put LocalOptimizer and Optimizer as EquilibriumOptimizer
    suppor both.

    """

    def __init__(self, ignore_newtrees=True):
        super(EquilibriumDB, self).__init__()
        self.ignore_newtrees = ignore_newtrees
        self.__final__ = {}
        self.__cleanup__ = {}

    def register(self, name, obj, *tags, **kwtags):
        final_opt = kwtags.pop('final_opt', False)
        cleanup = kwtags.pop('cleanup', False)
        # An opt should not be final and clean up
        assert not (final_opt and cleanup)
        super(EquilibriumDB, self).register(name, obj, *tags, **kwtags)
        self.__final__[name] = final_opt
        self.__cleanup__[name] = cleanup

    def query(self, *tags, **kwtags):
        _opts = super(EquilibriumDB, self).query(*tags, **kwtags)
        final_opts = [o for o in _opts if self.__final__.get(o.name, False)]
        cleanup_opts = [o for o in _opts if self.__cleanup__.get(o.name,
                                                                 False)]
        opts = [o for o in _opts
                if o not in final_opts and o not in cleanup_opts]
        if len(final_opts) == 0:
            final_opts = None
        if len(cleanup_opts) == 0:
            cleanup_opts = None
        return opt.EquilibriumOptimizer(
            opts,
            max_use_ratio=config.optdb.max_use_ratio,
            ignore_newtrees=self.ignore_newtrees,
            failure_callback=opt.NavigatorOptimizer.warn_inplace,
            final_optimizers=final_opts,
            cleanup_optimizers=cleanup_opts)


class SequenceDB(DB):
    """
    A sequence of potential optimizations.

    Retrieve a sequence of optimizations (a SeqOptimizer) by calling query().

    Each potential optimization is registered with a floating-point position.
    No matter which optimizations are selected by a query, they are carried
    out in order of increasing position.

    The optdb itself (`theano.compile.mode.optdb`), from which (among many
    other tags) fast_run and fast_compile optimizers are drawn is a SequenceDB.

    """

    seq_opt = opt.SeqOptimizer

    def __init__(self, failure_callback=opt.SeqOptimizer.warn):
        super(SequenceDB, self).__init__()
        self.__position__ = {}
        self.failure_callback = failure_callback

    def register(self, name, obj, position, *tags):
        super(SequenceDB, self).register(name, obj, *tags)
        assert isinstance(position, (integer_types, float))
        self.__position__[name] = position

    def query(self, *tags, **kwtags):
        """

        Parameters
        ----------
        position_cutoff : float or int
            Only optimizations with position less than the cutoff are returned.

        """
        opts = super(SequenceDB, self).query(*tags, **kwtags)

        position_cutoff = kwtags.pop('position_cutoff',
                                     config.optdb.position_cutoff)
        position_dict = self.__position__

        if len(tags) >= 1 and isinstance(tags[0], Query):
            # the call to super should have raise an error with a good message
            assert len(tags) == 1
            if getattr(tags[0], 'position_cutoff', None):
                position_cutoff = tags[0].position_cutoff

            # The Query instance might contain extra optimizations which need
            # to be added the the sequence of optimizations (don't alter the
            # original dictionary)
            if len(tags[0].extra_optimizations) > 0:
                position_dict = position_dict.copy()
                for extra_opt in tags[0].extra_optimizations:
                    # Give a name to the extra optimization (include both the
                    # class name for descriptiveness and id to avoid name
                    # collisions)
                    opt, position = extra_opt
                    opt.name = "%s_%i" % (opt.__class__, id(opt))

                    # Add the extra optimization to the optimization sequence
                    if position < position_cutoff:
                        opts.add(opt)
                        position_dict[opt.name] = position

        opts = [o for o in opts if position_dict[o.name] < position_cutoff]
        opts.sort(key=lambda obj: (position_dict[obj.name], obj.name))
        kwargs = {}
        if self.failure_callback:
            kwargs["failure_callback"] = self.failure_callback
        ret = self.seq_opt(opts, **kwargs)
        if hasattr(tags[0], 'name'):
            ret.name = tags[0].name
        return ret

    def print_summary(self, stream=sys.stdout):
        print(self.__class__.__name__ + " (id %i)" % id(self), file=stream)
        positions = list(self.__position__.items())

        def c(a, b):
            return ((a[1] > b[1]) - (a[1] < b[1]))
        positions.sort(c)

        print("  position", positions, file=stream)
        print("  names", self._names, file=stream)
        print("  db", self.__db__, file=stream)

    def __str__(self):
        sio = StringIO()
        self.print_summary(sio)
        return sio.getvalue()


class LocalGroupDB(SequenceDB):
    """
    Generate a local optimizer of type LocalOptGroup instead
    of a global optimizer.

    It supports the tracks, to only get applied to some Op.

    """

    seq_opt = opt.LocalOptGroup

    def __init__(self, failure_callback=opt.SeqOptimizer.warn):
        super(LocalGroupDB, self).__init__()
        self.failure_callback = None


class ProxyDB(DB):
    """
    Wrap an existing proxy.

    This is needed as we can't register the same DB mutiple times in
    different positions in a SequentialDB.

    """

    def __init__(self, db):
        assert isinstance(db, DB), ""
        self.db = db

    def query(self, *tags, **kwtags):
        return self.db.query(*tags, **kwtags)
