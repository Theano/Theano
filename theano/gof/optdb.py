
from collections import defaultdict
import opt


class DB(object):

    def __init__(self):
        self.__db__ = defaultdict(set)

    def register(self, name, obj, *tags):
        obj.name = name
        if name in self.__db__:
            raise ValueError('The name of the object cannot be an existing tag or the name of another existing object.', obj, name)
        self.__db__[name] = set([obj])
        for tag in tags:
            self.__db__[tag].add(obj)

    def __query__(self, q):
        if not isinstance(q, Query):
            raise TypeError('Expected a Query.', q)
        results = set()
        for tag in q.include:
            results.update(self.__db__[tag])
        for tag in q.require:
            results.intersection_update(self.__db__[tag])
        for tag in q.exclude:
            results.difference_update(self.__db__[tag])
        remove = set()
        add = set()
        for obj in results:
            if isinstance(obj, DB):
                sq = q.subquery.get(obj.name, q)
                if sq:
                    replacement = obj.query(sq)
                    replacement.name = obj.name
                    remove.add(obj)
                    add.add(replacement)
        results.difference_update(remove)
        results.update(add)
        return results

    def query(self, *tags, **kwtags):
        if len(tags) >= 1 and isinstance(tags[0], Query):
            if len(tags) > 1 or kwtags:
                raise TypeError('If the first argument to query is a Query, there should be no other arguments.', tags, kwtags)
            return self.__query__(tags[0])
        include = [tag[1:] for tag in tags if tag.startswith('+')]
        require = [tag[1:] for tag in tags if tag.startswith('&')]
        exclude = [tag[1:] for tag in tags if tag.startswith('-')]
        if len(include) + len(require) + len(exclude) < len(tags):
            raise ValueError("All tags must start with one of the following characters: '+', '&' or '-'", tags)
        return self.__query__(Query(include = include,
                                    require = require,
                                    exclude = exclude,
                                    subquery = kwtags))

    def __getitem__(self, name):
        results = self.__db__[name]
        if not results:
            raise KeyError("Nothing registered for '%s'" % name)
        elif len(results) > 1:
            raise ValueError('More than one match for %s (please use query)' % name)
        for result in results:
            return result


class Query(object):

    def __init__(self, include, require = None, exclude = None, subquery = None):
        self.include = include
        self.require = require or set()
        self.exclude = exclude or set()
        self.subquery = subquery or {}

    def including(self, *tags):
        return Query(self.include.union(tags),
                     self.require,
                     self.exclude,
                     self.subquery)

    def excluding(self, *tags):
        return Query(self.include,
                     self.require,
                     self.exclude.union(tags),
                     self.subquery)

    def requiring(self, *tags):
        return Query(self.include,
                     self.require.union(tags),
                     self.exclude,
                     self.subquery)




class EquilibriumDB(DB):

    def query(self, *tags, **kwtags):
        opts = super(EquilibriumDB, self).query(*tags, **kwtags)
        return opt.EquilibriumOptimizer(opts, max_depth = 5, max_use_ratio = 10, failure_callback = opt.warn)


class SequenceDB(DB):

    def __init__(self):
        super(SequenceDB, self).__init__()
        self.__priority__ = {}

    def register(self, name, obj, priority, *tags):
        super(SequenceDB, self).register(name, obj, *tags)
        self.__priority__[name] = priority

    def query(self, *tags, **kwtags):
        opts = super(SequenceDB, self).query(*tags, **kwtags)
        opts = list(opts)
        opts.sort(key = lambda obj: self.__priority__[obj.name])
        return opt.SeqOptimizer(opts, failure_callback = opt.warn)


