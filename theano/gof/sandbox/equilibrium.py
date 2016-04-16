from __future__ import absolute_import, print_function, division

from six.moves import reduce
from six import string_types

if 0:
    class _EquilibriumOptimizer(NavigatorOptimizer):

        def __init__(self,
                     local_optimizers,
                     failure_callback=None,
                     max_depth=None,
                     max_use_ratio=None):

            super(EquilibriumOptimizer, self).__init__(
                None,
                ignore_newtrees=False,
                failure_callback=failure_callback)

            self.local_optimizers = local_optimizers
            self.max_depth = max_depth
            self.max_use_ratio = max_use_ratio

            self.tracks = defaultdict(list)
            self.tracks0 = defaultdict(list)
            max_depth = 0
            for lopt in local_optimizers:
                tracks = lopt.tracks()
                for track in tracks:
                    max_depth = max(max_depth, len(track))
                    if self.max_depth is not None and max_depth > self.max_depth:
                        raise ValueError('One of the local optimizers exceeds the maximal depth.')
                    for i, op in enumerate(track):
                        if i == 0:
                            self.tracks0[op].append((track, i, lopt))
                        self.tracks[op].append((track, i, lopt))

        def fetch_tracks(self, op):
            return self.tracks[op] + self.tracks[None]

        def fetch_tracks0(self, op):
            return self.tracks0[op] + self.tracks0[None]

        def backtrack(self, node, tasks):
            candidates = self.fetch_tracks(node.op)
            tracks = []
            def filter(node, depth):
                new_candidates = []
                for candidate in candidates:
                    track, i, lopt = candidate
                    if i < depth:
                        pass
                    elif track[i-depth] in (None, node.op):
                        if i == depth:
                            tasks[node].append(lopt)
                        else:
                            tracks.append(candidate)
                    else:
                        new_candidates.append(candidate)
                return new_candidates
            depth = 0
            nodes = [node]
            while candidates:
                for node in nodes:
                    candidates = list(filter(node, depth))
                depth += 1
                _nodes = nodes
                nodes = reduce(list.__iadd__,
                               [reduce(list.__iadd__,
                                       [[n for n, i in out.clients if not isinstance(n, string_types)] for out in node.outputs],
                                       []) for node in nodes],
                               [])
                candidates = tracks
                tracks = []

        def apply(self, fgraph):
            tasks = defaultdict(list)

            if self.max_use_ratio is not None:
                max_uses = self.max_use_ratio * len(fgraph.apply_nodes)
                runs = defaultdict(int)
            else:
                runs = None

            def importer(node):
                # print 'IMPORTING', node
                self.backtrack(node, tasks)
            def pruner(node):
                try:
                    del tasks[node]
                except KeyError:
                    pass
            def chin(node, i, r, new_r):
                if new_r.owner and not r.clients:
                    self.backtrack(new_r.owner, tasks)

    #         # == NOT IDEAL == #
    #         for node in fgraph.apply_nodes:
    #             importer(node)

            for node in fgraph.toposort():
                tasks[node].extend(lopt for track, i, lopt in self.fetch_tracks0(node.op))

            u = self.attach_updater(fgraph, importer, pruner, chin)
            print('KEYS', [hash(t) for t in tasks.keys()])
            while tasks:
                for node in tasks:
                    todo = tasks.pop(node)
                    break
                for lopt in todo:
                    if runs is not None and runs[lopt] >= max_uses:
                        print('Warning: optimization exceeded its maximal use ratio: %s, %s' % (lopt, max_uses), file=sys.stderr)
                        continue
                    success = self.process_node(fgraph, node, lopt)
                    if success:
                        if runs is not None: runs[lopt] += 1
                        break
            self.detach_updater(fgraph, u)

#     def match(self, node, candidates):
#         candidates[:] = [candidate
#                          for candidate in candidates
#                          if candidate.current.op is None or candidate.current.op == node.op]
#         for candidate in candidates:
#             if candidate.current.inputs is not None:
#                 for in1, in2 in zip(candidate.current.inputs, node.inputs):
#                     if isinstance(in1, string_types):
#                         candidate.match[in1] = in2
#         for client in node.clients:


#         op = node.op
#         patterns = self.pattern_base[(depth, op)].union(self.pattern_base[(depth, WILDCARD)])
#         if not patterns:
#             return patterns
#         return self.match(node, depth + 1).intersection(patterns)


#     def backtrack(self, node, q):
#         for node2, i in node.clients:
#             op2 = node2.op





