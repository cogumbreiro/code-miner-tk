#!/usr/bin/env python3
import sys

try:
    import graphviz
except ImportError as e:
    print("ERROR:", e.args[0], file=sys.stderr)
    print("\nInstal the needed dependencies with:\n\tpython3 -m pip install graphviz", file=sys.stderr)
    sys.exit(1)

import warnings
import math
import argparse
import numpy as np
import itertools
from operator import *
import os
import errno
import weakref
import collections
import functools
import string
import cmd
import shlex

# Shut up Tensorflow
if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

import common
import sal


# Decorator related

def memoize(fun):
    return functools.lru_cache(maxsize=None)(fun)

def as_list(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return list(f(*args, **kwargs))
    return wrapper

def cons_last(iterable, elem):
    yield from iterable
    yield elem


# Stats:

def ideal_similarity(vec1):
    return cosine_similarity(vec1, np.ones(len(vec1)))

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def three_sigma(arr):
    three_std = np.std(arr) * 3
    mu = arr.mean()
    def result(x):
        return abs(x - mu)/ three_std
    return result


def low_pass(x, lower_bound):
    return x if x > lower_bound else lower_bound

def low_pass_filter(elems, lower_bound):
    return (low_pass(x, lower_bound) for x in elems)

def low_pass_log(x, lower_bound=1e-40):
    return math.log(low_pass(x, lower_bound))

######

def group_pairs_by_key(pairs):
    """
    Takes a generator of (key*values) and groups the values, yielding a generator
    of (key * generator(values)).
    """
    # elems: list(loc*prob)
    elems = sorted(pairs, key=itemgetter(0))

    # elems: list(location * list(location*prob))
    elems = itertools.groupby(elems, key=itemgetter(0))

    # elems: list(location * list probs)
    return ((k, map(itemgetter(1), row)) for (k, row) in elems)

def dip_likelihood(likelihoods):
    """
    Given an array of likelihoods, returns a score that takes into account
    vectors that are very similar 
    """
    arr = np.fromiter(common.skip_n(likelihoods, 1), np.float64)
    smallest = 1 - arr.min()
    return (arr.mean() ** 2 + smallest ** 2) / 2

def mean_log_likelihood(sequences, average_result=True):
    # XXX: We do not handle repeated sequences, as it is very expensive to
    # identify them; additionally we do not add log(1/n), as this value is
    # negligible, regardless of how big sequences go
    seqs = list(sequences)
    N = len(seqs)
    kld = np.zeros(N, np.float64)
    getter = attrgetter("state_probs" if average_result else "state_probs_cumulative")
    for idx, seq in enumerate(seqs):
        kld[idx] = getter(seq)
    
    return -low_pass_log(kld.prod()) / N


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

    def group_by_location(self, get_probs, on_path, min_length=3):
        """
        Returns a generator where the key is the location and the value
        is a generator of probabilities (max call likelihood)
        """
        seqs = filter(lambda x:len(x) >= min_length, self)
        def probs():
            known = set()
            for seq in seqs:
                path_id = ""
                path = []
                for idx, (call, prob) in enumerate(zip(seq, get_probs(seq))):
                    path.append(prob)
                    path_id += "/" + call.call
                    if idx < min_length:
                        continue
                    if path_id not in known:
                        # ensure we don't compute this twice
                        known.add(path_id)
                        yield call.location, on_path(path)

        return group_pairs_by_key(probs())

    def group_by_dip(self):
        """
        Takes the max likelihood
        """
        probs = self.group_by_location(
            get_probs=ASequence.get_max_call_likelihood,
            on_path=dip_likelihood
        )
        return ((x, max(scores)) for x,scores in probs)

    def group_by_log_likelihood(self, average_result, aggr=attrgetter("mean")):
        #elems = self.group_by_last_location()
        #return ((l, mean_log_likelihood(s, average_result=average_result)) for l, s in elems)
        if average_result:
            def per_call(x):
                return np.fromiter(x[1], np.float64).prod() / (x[0] + 1)
        else:
            def per_call(x):
                return np.fromiter(x[1], np.float64).prod()
        def get_probs(seq):
            return map(per_call, enumerate(seq.get_state_probs()))
        
        probs = self.group_by_location(
            get_probs=get_probs,
            on_path=lambda x: -low_pass_log(np.fromiter(common.skip_n(x, 1), np.float64).prod())
        )
        return ((x, aggr(np.fromiter(scores, np.float64))) for x,scores in probs)

StateProbs = collections.namedtuple('StateProbs', ['value', 'states', 'max'])

class ASequence(sal.VSequence):
    def __init__(self, js, sid, spec, parent):
        self.js = js
        self.sid = sid
        self.spec = spec
        self.parent = weakref.ref(parent)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ASequence(
                js = {'sequence': self.js['sequence'][key]},
                sid = self.sid,
                spec = self.spec,
                parent = self.parent(),
            )
        else:
            return super(ASequence, self).__getitem__(key)

    def subsequences(self, predicate:lambda x: True, min_length=3, max_length=-1):
        visited = set()
        for idx, call in enumerate(self):
            last_idx = idx + 1
            if predicate(call):
                for start_idx in range(last_idx):
                    seq_len = last_idx - start_idx
                    if seq_len >= min_length and (max_length == -1 or seq_len <= max_length):
                        seq = self[start_idx:last_idx]
                        seq_id = seq.as_string(include_location=False)
                        if seq_id not in visited:
                            visited.add(seq_id)
                            yield seq

    def call_dist(self):
        js_events = sal.get_calls(seq=self.js)
        app = self.parent()
        return app.distribution_call_iter(self.spec, js_events, cache=app.cache)

    def next_calls(self):
        return cons_last((c.call for c in self), sal.END_MARKER)        

    @memoize
    @as_list
    def get_state_probs(self):
        """
        Returns the join probability of all next-calls and the number of
        probabilities counted.
        """
        js_events = sal.get_calls(seq=self.js)
        app = self.parent()
        state_dist = app.distribution_state_iter(self.spec, js_events, cache=app.cache)
        for next_call, row in zip(self.next_calls(), state_dist):
            elems = [row.distribution[next_call]]
            for prob in row.states:
                elems.append(prob)
            if next_call != sal.END_MARKER:
                dist = row.next_state()
                elems.append(dist[sal.END_MARKER])
            yield elems

    @memoize
    @as_list
    def get_state_probs_ex(self):
        """
        Returns the join probability of all next-calls and the number of
        probabilities counted.
        """
        js_events = sal.get_calls(seq=self.js)
        app = self.parent()
        state_dist = app.distribution_state_iter(self.spec, js_events, cache=app.cache)
        for next_call, row in zip(self.next_calls(), state_dist):
            if next_call == sal.end.MARKER:
                states = []
            else:
                states = list(row.states)
                states.append(row.next_state()[sal.END_MARKER])

            yield StateProbs(
                value=row.distribution[next_call],
                states=states,
                max=lambda: max(row.distribution.values())
            )

    def get_state_probs(self):
        """
        Returns the join probability of all next-calls and the number of
        probabilities counted.
        """
        def on_elem(x):
            row = [x.value]
            row.extend(x.states)
            return row

        return map(on_elem, self.get_state_probs_ex())

    def state_probs(self, count=None):
        if count is None:
            count = len(self)
        elems = itertools.chain.from_iterable(self.get_state_probs()[0:count])
        arr = np.fromiter(elems, np.float64)
        return arr.prod() / len(arr)

    def state_probs_cumulative(self, count=None):
        if count is None:
            count = len(self)
        elems = itertools.chain.from_iterable(self.get_state_probs()[0:count])
        arr = np.fromiter(elems, np.float64)
        return arr.prod()

    log = property(lambda x: -low_pass_log(x.state_probs()))
    log_cumulative = property(lambda x: -low_pass_log(x.state_probs_cumulative()))

    @memoize
    @as_list
    def get_max_call_likelihood(self):
        """
        Returns the call likelihood at each position of the sequence.
        The likelihood is the probability of the call divided by the probability
        of the most probable call.
        """
        for row, next_call in zip(self.call_dist(), self.next_calls()):
            dist = row.distribution
            biggest = max(dist.values())
            yield dist[next_call] / biggest

    @memoize
    def ideal_likelihood(self, log_scale=True, average_result=True):
        curr = np.zeros(len(self) + 1, dtype=np.float64)
        for idx, ratio in enumerate(self.get_max_call_likelihood()):
            curr[idx] = ratio

        if log_scale:
            np.log(curr, curr)
            result = -curr.sum()
        else:
            result = curr.sum()
        
        return result / len(curr) if average_result else result

    @property
    @memoize
    def dip(self):
        arr = np.array(self.get_max_call_likelihood())
        return dip_likelihood(arr)


    ideal = property(lambda x: x.ideal_likelihood(log_scale=False, average_result=True))
    ideal_log = property(lambda x: x.ideal_likelihood(log_scale=True, average_result=True))
    ideal_log_cumulative = property(lambda x: x.ideal_likelihood(log_scale=True, average_result=True))

    def show(self):
        node = "{}".format
        dists = map(attrgetter("distribution"), self.call_dist())
        is_first = True
        for event, dist, next_call in zip(cons_last(self, None), dists, self.next_calls()):
            highest_key, highest = max(dist.items(), key=lambda x:x[1])
            ratio = dist[next_call] / highest

            label = next_call
            if event is not None:
                label += ":" + event.location

            if is_first or highest_key == next_call or ratio > .2:
                # Skip showing the anomaly score
                _1_col = "    "
                _2_col = ""
            else:
                _1_col = "{0:4.0%}".format(float(ratio))
                _2_col = "\t\texpecting: {0:4.0%} {1} ".format(highest, highest_key)

            print(_1_col, label, _2_col)
            is_first = False

        print(". " * 40)
        print()

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

#################
# User Interface

def make_app(*args, **kwargs):
    from salento.aggregators.base import Aggregator

    class App(Aggregator):

        """
        The simple sequence aggregator computes, for each sequence, the negative
        log-likelihood of the sequence using only its calls (not states).
        """
        def __init__(self, data_file, model_dir):
            Aggregator.__init__(self, data_file, model_dir)
            self.cache = {}

        def init(self):
            self.pkgs = ADataset(self.dataset, self)
            # Remove calls that are not in the vocab
            unknown = set()
            def on_unknown(term):
                call = term['call']
                if call not in unknown:
                    unknown.add(call)
                    print("UNKNOWN CALL", call)
            call_filter = lambda f: sal.make_filter_on_reject(f, on_unknown)
            self.pkgs.filter_calls(vocabs=self.model.model.config.decoder.vocab, call_filter=call_filter)
            # No need to show sequences with only 2 terms or less
            self.pkgs.filter_sequences(min_length=3)

        def log(self, *args, **kwargs):
            pass


    return App(*args, **kwargs)

def parse_line(fun):
    def wrapper(self, line):
        name = fun.__name__[3:]
        parser = argparse.ArgumentParser(description=fun.__doc__, prog=name)
        parser.exit = self.error
        getattr(self, 'argparse_' + name)(parser)
        try:
            try:
                args = shlex.split(line)
            except ValueError as e:
                raise REPLExit("Error parsing arguments of command %r: %s" % (name, e))
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
    return list(map(common.parse_slice, expr.split(",")))

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

    ALGOS = {
        'mean-ll': lambda pkg, args: pkg.group_by_log_likelihood(average_result=args.average, aggr=attrgetter("mean")),
        'max-ll': lambda pkg, args: pkg.group_by_log_likelihood(average_result=args.average, aggr=attrgetter("max")),
        'max-min': lambda pkg, args: pkg.group_by_dip(),
    }

    def argparse_group(self, parser):
        # Filter which packages.
        parser.add_argument('--pid', default='*', help="A query to match packages, the format is a Python slice expression, so ':' retreives all packages in the dataset. You can also use '*' to match all elements. Default: %(default)r")
        parser.add_argument("--fmt", "-f", default='pid: {pkg.pid} pkg: {pkg.name} by: {last_location} anomaly: {score:.0%}', help='Print format. Default: %(default)s')
        parser.add_argument("--fmt-extra", "-p", nargs='*', default='', help='Append format. Default: %(default)s')
        parser.add_argument("--reverse", action="store_false", help="Reverese the order of the results.")
        parser.add_argument('--limit', default=-1, type=int, help="Limit the number of elements shown *per package*.")
        parser.add_argument("--no-sort", dest="sort", action="store_false", help='By default we sort the values by their KLD value; this switch disables sorting.')
        parser.add_argument('--no-avg', dest='average', action='store_false',
            help='By default divide the score by the length of the sequence. This flag disables this step.') 
        parser.add_argument('--algo', default='max-min', choices=self.ALGOS.keys())
        parser.add_argument('--filter')

    @parse_line
    def do_group(self, args):
        """
        Run on all packages, grouped by the last location.
        """
        app = self.app
        try:
            pkg_ids = parse_ranges(args.pid)
            filter_elems = None
            if args.filter is not None:
                try:
                    filter_elems = eval(args.filter)
                except (BaseException,TypeError) as e:
                    raise ValueError(e)
        except ValueError as e:
            raise REPLExit("Error parsing pkg-ids %r:" % args.pid, str(e))
        
        for pkg in app.pkgs.lookup(pkg_ids):
            elems = self.ALGOS[args.algo](pkg, args)
            if filter_elems is not None:
                elems = filter(lambda x: filter_elems(x[1]), elems)
            if args.sort:
                elems = sorted(elems, key=lambda x:x[1], reverse=args.reverse)
            if args.limit > -1:
                elems = common.take_n(elems, args.limit)
            for l, r in elems:
                fmt = args.fmt
                fmt += "".join(args.fmt_extra)
                #max_seq_len = max(map(len, s))
                print(repl_format(fmt, pkg=pkg, last_location=l, score=r))

    def argparse_seq(self, parser):
        # Filter which packages.
        parser.add_argument('--pid', default='*', help="A query to match packages, the format is a Python slice expression, so ':' retreives all packages in the dataset. You can also use '*' to match all elements.")
        parser.add_argument('--sid', default='*', help="A query to select sequences, by default we match all ids. You can use '*' to match all sequences.")
        # Message
        parser.add_argument('--fmt', '-f', default='pid: {pkg.pid} sid: {seq.sid} count: {seq.count} last: {seq.last_location} anomalous: {seq.dip:.0%}', help="Default: %(default)r")
        parser.add_argument("--fmt-extra", "-p", nargs='*', default='', help='Append format. Default: %(default)s')
        # Limit output
        parser.add_argument('--limit', default=-1, type=int, help="Limit the number of elements shown.")
        parser.add_argument('--unique', action='store_true', help="Only show only unique sequences.")
        # Save visualization
        parser.add_argument('--print', action='store_true', help='Visualize the trace on the screen.')
        parser.add_argument('--save', action='store_true', help='Write the visualization to a filename.')
        parser.add_argument('--save-fmt', default="{pkg.pid}-{seq.sid}{sid_extra}.gv", help='Visualization format. Default: %(default)r')
        # Queries to filter sequences
        parser.add_argument('--start', '-s', help='Filter in sequences that start with the given location.')
        parser.add_argument('--end', '-e', help='Filter in sequences that end with the given location.')
        parser.add_argument('--match', '-m', help='Filter in sequences that contain the given location.')
        parser.add_argument('--sub', help='Sub-sequences ending in the given location')
        parser.add_argument('--subs', action="store_true", help='Range over all sub-sequences')
        # Sort the final list
        parser.add_argument('--sort', default='dip', choices=["log", "ideal", "ideal_log", "sid", 'dip'], help='Sorts the output by a field')
        parser.add_argument('--reverse', '-r', action='store_false')
        parser.add_argument('--min-length', default=3, type=int, help='The minimum size of a call sequence; anything below is ignored. Default: %(default)r')
        parser.add_argument('--max-length', default=-1, type=int, help='The maximum size of a call sequence; anything above is ignored. Value -1 disables this check. Default: %(default)r')

    @parse_line
    def do_seq(self, args):
        """
        Run queries at the sequence level.
        """

        app = self.app
        try:
            pkg_ids = parse_ranges(args.pid)
            seq_ids = parse_ranges(args.sid) if args.sid is not None else None
        except ValueError as e:
            raise REPLExit("Error parsing pkg-ids %r:" % args.pid, str(e))

        get_location = attrgetter("location")
        
        if args.unique:
            visited = set()
        for pkg in app.pkgs.lookup(pkg_ids):
            if args.sub is not None or args.subs:
                if args.subs:
                    do_filter = lambda x: True
                else:
                    do_filter = lambda x: sal.match(x.location, args.sub)
                elems = (seq.subsequences(do_filter, min_length=args.min_length, max_length=args.max_length) for seq in pkg)
                elems = itertools.chain.from_iterable(elems)
                elems = set(elems)
            else:
                elems = pkg
            elems = filter(lambda seq: len(seq) >= args.min_length and \
                (args.max_length == -1 or len(seq) <= args.max_length), elems)
            
            # Only show unique elements
            if args.unique:
                new_elems = []
                for x in elems:
                    x_id = x.as_string(include_location=False)
                    if x_id not in visited:
                        new_elems.append(x)
                        visited.add(x_id)
                elems = new_elems

            if args.sort is not None:
                elems = sorted(elems, key=attrgetter(args.sort), reverse=args.reverse)

            if args.limit >= 0:
                elems = common.take_n(elems, args.limit)

            if seq_ids is not None:
                accept = set()
                for sids in seq_ids:
                    for elem in range(*sids.indices(len(pkg))):
                        accept.add(elem)
                elems = filter(lambda x: x.sid in accept, elems)
            counter = collections.Counter()
            for seq in elems:
                counter[seq.sid] += 1
                if args.end is not None and not seq.matches_at(args.end, -1, get_location):
                    continue
                if args.match is not None and not seq.matches_any(args.match, get_location):
                    continue
                if args.start is not None and not seq.matches_at(args.start, 0, get_location):
                    continue

                sid_extra = str(counter[seq.sid]) if counter[seq.sid] > 1 else ""

                def do_fmt(x):
                    return repl_format(x, pkg=pkg, seq=seq, sid_extra=sid_extra)

                if args.save:
                    fname = do_fmt(args.save_fmt)
                    g = graphviz.Digraph(comment=pkg.name, filename=fname)
                    seq.visualize(g)
                    g.save()
                elif args.print:
                    seq.show()
                else:
                    fmt = args.fmt
                    fmt += "".join(args.fmt_extra)
                    print(do_fmt(fmt))



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
        
