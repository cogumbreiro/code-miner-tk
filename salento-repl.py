#!/usr/bin/env python3
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

try:
    import sal
except ImportError:
    import sys
    import os
    from os import path
    home = path.abspath(path.dirname(sys.argv[0]))
    sys.path.append(path.join(home, "src"))

import common
import sal
import math
import argparse
import numpy as np
import itertools
from operator import *
import graphviz
import os
import errno
import weakref
import collections

def probs_cosine_similarity(vec1):
    return cosine_similarity(vec1, np.ones(len(vec1)))

def cosine_similarity(vec1, vec2):
    assert len(vec1) == len(vec2)
    if len(vec1) == 0:
        return 1.0
    return vec1.dot(vec2) / (np.linalg.norm(vec1, ord=1)*np.linalg.norm(vec2, ord=1))


class ADataset(sal.VDataset):
    def __init__(self, js, parent):
        self.js = js
        self.parent = weakref.ref(parent)

    def make_package(self, js, pid):
        parent = self.parent()
        spec = parent.get_latent_specification(js)
        return APackage(js, pid, spec, parent)

    def lookup(self, pkg_ids):
        for ids in pkg_ids:
            yield from self[ids]

class APackage(sal.VPackage):
    def __init__(self, js, pid, spec, parent):
        self.js = js
        self.pid = pid
        self.spec = spec
        self.parent = weakref.ref(parent)

    def make_sequence(self, js, sid):
        return ASequence(js, sid, self.spec, self.parent())

    def lookup(self, seq_ids):
        for ids in seq_ids:
            yield from self[ids]


def cons_last(iterable, elem):
    yield from iterable
    yield elem

class ASequence(sal.VSequence):
    def __init__(self, js, sid, spec, parent):
        self.js = js
        self.sid = sid
        self.spec = spec
        self.parent = weakref.ref(parent)

    def state_dist(self):
        js_events = sal.get_calls(seq=self.js)
        app = self.parent()
        return app.distribution_state_iter(self.spec, js_events, cache=app.cache)

    def call_dist(self):
        js_events = sal.get_calls(seq=self.js)
        app = self.parent()
        return app.distribution_call_iter(self.spec, js_events, cache=app.cache)

    def call_names(self):
        return (c.call for c in self)

    def next_calls(self):
        return cons_last(self.call_names(), sal.END_MARKER)        

    def log_likelihood(self, average_result=True):
        llh = 0.
        count = 0
        for next_call, row in zip(self.next_calls(), self.state_dist()):
            llh += math.log(row.distribution[next_call])
            count += 1
            for prob in row.states:
                llh += math.log(prob)
                count += 1
            if next_call != sal.END_MARKER:
                dist = row.next_state()
                llh += math.log(dist[sal.END_MARKER])
                count += 1
        return llh / count if average_result else llh

    log = property(lambda x: -x.log_likelihood(average_result=False))
    log_cumulative = property(lambda x: -x.log_likelihood(average_result=True))

    def ideal_likelihood(self, log_scale=True, average_result=True):
        curr = np.zeros(len(self) + 1, dtype=np.float64)
        for (idx, (row, next_call)) in enumerate(zip(self.call_dist(), self.next_calls())):
            # Here we check how far each node is from the optimal choice
            #dist = ((k,v) for (k,v) in row.distribution.items() if k != 'START' and '#' not in k)
            dist = row.distribution
            biggest = max(dist.values())
            curr[idx] = dist[next_call] / biggest

        if log_scale:
            np.log(curr, curr)
            result = -curr.sum()
        else:
            result = curr.sum()
        
        return result / len(curr) if average_result else result

    ideal = property(lambda x: x.ideal_likelihood(log_scale=False, average_result=True))
    ideal_log = property(lambda x: x.ideal_likelihood(log_scale=True, average_result=True))
    ideal_log_cumulative = property(lambda x: x.ideal_likelihood(log_scale=True, average_result=True))

    def visualize(self, g):
        last_call = None
        node_id = 0
        node = "{}".format
        g.node(node(node_id), label="START")
        dists = map(attrgetter("distribution"), self.call_dist())
        for event, dist, next_call in zip(cons_last(self, None), dists, self.next_calls()):
            label = next_call
            if event is not None:
                label += ":" + event.location
            
            highest_key, highest = max(dist.items(), key=lambda x:x[1])
            ratio = dist[next_call] / highest
            color = "#%x4900" % int(256 - 256 * ratio)
            g.node(node(node_id+1), label=label, color=color)
            kwargs = {}
            if highest_key != next_call:
                kwargs["label"] = "%0.2f" % ratio

            g.edge(node(node_id), node(node_id+1), color=color, **kwargs)
            if highest_key != next_call:
                max_node = node(node_id+1) + "_max"
                g.node(max_node, label=highest_key)
                g.edge(node(node_id), max_node, label="%0.2f" % highest)
            node_id += 1

def compute_kld(sequences, average_result=True):
    # XXX: we do not handle repeated sequences, as it is very expensive to identify them
    elems = list(sequences)
    total = len(elems)
    kld = 0.
    for sequence in elems:
        p = 1 / total
        log_p = math.log(p)
        log_q = sequence.log_likelihood(average_result=average_result)
        kld += p * (log_p - log_q)
    return kld

def make_app(*args, **kwargs):
    from salento.aggregators.base import Aggregator

    class App(Aggregator):
        filter_unknown_vocabs = False

        """
        The simple sequence aggregator computes, for each sequence, the negative
        log-likelihood of the sequence using only its calls (not states).
        """
        def __init__(self, data_file, model_dir):
            Aggregator.__init__(self, data_file, model_dir)
            self.cache = {}

        def init(self):
            sal.filter_unknown_vocabs(self.dataset, self.model.model.config.decoder.vocab)
            self.pkgs = ADataset(self.dataset, self)

        def log(self, *args, **kwargs):
            pass


    return App(*args, **kwargs)

import cmd
import shlex

def parse_line(fun):
    def wrapper(self, line):
        name = fun.__name__[3:]
        parser = argparse.ArgumentParser(description=fun.__doc__, prog=name)
        parser.exit = self.error
        getattr(self, 'argparse_' + name)(parser)
        try:
            try:
                args = shlex.split(line)
            except ValueError:
                raise REPLExit("Error parsing arguments of command %r: %s" (name, e))
            fun(self, parser.parse_args(args))
        except REPLExit as e:
            print(e)
        except KeyboardInterrupt:
            pass
    wrapper.__name__ = fun.__name__
    wrapper.__doc__ = fun.__doc__
    return wrapper

class REPLExit(Exception):
    pass

def take_n(iterable, count):
    for x, _ in zip(iterable, range(count)):
        yield x

import string
class CallFormatter(string.Formatter):
    def format_field(self, value, spec):
        if spec == 'call':
            return value()
        else:
            return super(CallFormatter, self).format_field(value, spec)


def parse_ranges(expr):
    expr = expr.strip()
    if expr == '' or expr == '*':
        return [common.parse_slice(":")]
    return map(common.parse_slice, expr.split(","))

def repl_format(*args, **kwargs):
    fmt = CallFormatter()
    try:
        return fmt.format(*args, **kwargs)
    except (TypeError, KeyError, ValueError, AttributeError) as e:
        raise REPLExit("Error parsing format: %s" % e)

class REPL(cmd.Cmd):
    prompt = '> '
    intro = 'Welcome to the Salento shell. Type help or ? to list commands.\n'
    def __init__(self, app):
        cmd.Cmd.__init__(self)
        self.app = app


    def do_pkgs(self, line):
        """
        Lists all packages.
        """
        for pkg in self.app.pkgs:
            print(pkg.name)

    def error(self, error_code=2, msg=None):
        if msg is not None:
            print(msg, file=sys.stderr)
        raise REPLExit

    def argparse_kld(self, parser):
        # Filter which packages.
        parser.add_argument('pkg_id', default='*', nargs='?', help="A query to match packages, the format is a Python slice expression, so ':' retreives all packages in the dataset. You can also use '*' to match all elements. Default: %(default)r")
        parser.add_argument("--fmt", "-f", default='id: {pkg.pid} location: {pkg.name} | {last_location} score: {score:.1f}', help='Print format. Default: %(default)s')
        parser.add_argument("--reverse", action="store_false", help="Reverese the order of the results.")
        parser.add_argument('--limit', default=-1, type=int, help="Limit the number of elements shown.")
        parser.add_argument("--no-sort", dest="sort", action="store_false", help='By default we sort the values by their KLD value; this switch disables sorting.')
        parser.add_argument('--no-avg', dest='average', action='store_false',
            help='By default divide the score by the length of the sequence. This flag disables this step.') 

    @parse_line
    def do_kld(self, args):
        """
        Run KLD on all packages.
        """
        app = self.app
        try:
            pkg_ids = parse_ranges(args.pkg_id)
        except ValueError as e:
            raise REPLExit("Error parsing pkg-ids %r:" % args.pkg_id, str(e))
        for pkg in app.pkgs.lookup(pkg_ids):
            elems = ((l, compute_kld(s, average_result=args.average)) for l, s in pkg.group_by_last_location())
            if args.sort:
                elems = sorted(elems, key=lambda x:x[1], reverse=args.reverse)
            if args.limit > -1:
                elems = take_n(elems, args.limit)
            for l, r in elems:
                print(repl_format(args.fmt, pkg=pkg, last_location=l, score=r))

    def argparse_seq(self, parser):
        # Filter which packages.
        parser.add_argument('pkg_id', help="A query to match packages, the format is a Python slice expression, so ':' retreives all packages in the dataset. You can also use '*' to match all elements.")
        parser.add_argument('seq', help="A query to select sequences, by default we match all ids. You can use '*' to match all sequences.")
        # Message
        parser.add_argument('--fmt', '-f', default='id: {seq.sid} count: {seq.count} last: {seq.last_location}', help="Default: %(default)r")
        # Limit output
        parser.add_argument('--limit', default=-1, type=int, help="Limit the number of elements shown.")
        # Save visualization
        parser.add_argument('--viz', action='store_true', help='Write the visualization to a filename.')
        parser.add_argument('--viz-fmt', default="{pkg.pid}-{seq.sid}.gv", help='Visualization format. Default: %(default)r')
        # Queries to filter sequences
        parser.add_argument('--start', '-s', help='Filter in sequences that start with the given location.')
        parser.add_argument('--end', '-e', help='Filter in sequences that end with the given location.')
        parser.add_argument('--match', '-m', help='Filter in sequences that contain the given location.')
        # Sort the final list
        parser.add_argument('--sort', choices=["log", "ideal", "ideal_log", "id"], help='Sorts the output by a field')
        parser.add_argument('--reverse', '-r', action='store_true')
        parser.add_argument('--min-length', default=3, type=int, help='The minimum size of a call sequence; anything below is ignored. Default: %(default)r')

    @parse_line
    def do_seq(self, args):
        """
        Run queries at the sequence level.
        """

        app = self.app
        try:
            pkg_ids = parse_ranges(args.pkg_id)
            seq_ids = parse_ranges(args.seq) if args.seq is not None else None
        except ValueError as e:
            raise REPLExit("Error parsing pkg-ids %r:" % args.pkg_id, str(e))

        get_location = attrgetter("location")

        for pkg in app.pkgs.lookup(pkg_ids):
            elems = filter(lambda seq: len(seq) >= args.min_length, pkg)
            if args.sort is not None:
                elems = sorted(elems, key=attrgetter(args.sort), reverse=args.reverse)

            if args.limit >= 0:
                elems = take_n(elems, args.limit)

            if seq_ids is not None:
                accept = set()
                for sids in seq_ids:
                    for elem in range(*sids.indices(len(pkg))):
                        accept.add(elem)
                elems = filter(lambda x: x.sid in accept, elems)

            for seq in elems:
                if args.end is not None and not seq.matches_at(args.end, -1, get_location):
                    continue
                if args.match is not None and not seq.matches_any(args.match, get_location):
                    continue
                if args.start is not None and not seq.matches_at(args.start, 0, get_location):
                    continue
                

                def do_fmt(x):
                    return repl_format(x, pkg=pkg, seq=seq)

                if args.viz:
                    fname = do_fmt(args.viz_fmt)
                    g = graphviz.Digraph(comment=pkg.name, filename=fname)
                    seq.visualize(g)
                    g.save()
                else:                
                    print(do_fmt(args.fmt))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input data file')
    parser.add_argument('--dirname', '-d', default="save",
                        help='directory to load model from')
    args = parser.parse_args()

    with make_app(args.filename, args.dirname) as aggregator:
        aggregator.init()
        repl = REPL(aggregator)
        repl.cmdloop()
    
if __name__ == '__main__':
    main()
        
