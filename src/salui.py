from typing import *
import numpy as np
import weakref

import sal
import statedist
from common import as_list

memoize = lambda x: x

class ADataset(sal.Dataset):
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

class APackage(sal.Package):
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

class ASequence(sal.Sequence):
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

    @memoize
    @as_list
    def get_call_probs(self):
        js_events = sal.get_calls(seq=self.js)
        app = self.parent()
        return app.distribution_call_iter(self.spec, js_events, cache=app.cache)

    @memoize
    @as_list
    def get_state_probs(self):
        """
        Returns the join probability of all next-calls and the number of
        probabilities counted.
        """
        js_events = sal.get_calls(seq=self.js)
        app = self.parent()
        for row in app.distribution_state_iter(self.spec, js_events, cache=app.cache):
            yield app.dist_adapter(row)

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
            vocab_set = self.model.model.config.decoder.vocab
            self.pkgs.filter_vocabs(vocabs=vocab_set, call_filter=call_filter)
            chars = self.model.model.config.decoder.chars
            self.dist_adapter = statedist.create_adapter(np.array(chars))

        def log(self, *args, **kwargs):
            # Ensure we are not printing out any debugging info
            pass


    return App(*args, **kwargs)

class EmptyCounter:
    def __getitem__(self, key):
        return 0

EmptyCounter = EmptyCounter()

class StateVar:
    """
    >>> sv = StateVar()
    >>> sv.calls
    {}
    >>> sv.get_freq('foo', '1')
    0
    >>> sv.inc_freq('foo', '1')
    >>> sv.inc_freq('foo', '1')
    >>> sv.inc_freq('foo', '1')
    >>> sv.get_freq('foo', '1')
    3

    >>> sv.get_freq('bar', '1')
    0
    >>> sv.inc_freq('bar', '1')
    >>> sv.inc_freq('bar', '1')
    >>> sv.get_freq('bar', '1')
    2

    >>> sv.inc_freq('baz', '1')
    >>> sv.calls
    {'foo': Counter({'1': 3}), 'bar': Counter({'1': 2}), 'baz': Counter({'1': 1})}

    Check max frequency:
    >>> sv.update_max_freq()
    >>> sv.max_freqs
    Counter({'1': 3})

    >>> sv.get_weight('foo', '1')
    1.0
    >>> sv.get_weight('bar', '1') == 2/3
    True
    >>> sv.get_weight('baz', '1') == 1/3
    True
    >>> 
    >>> elems = list((x, list(y) ) for x, y in sv.build())
    >>> elems == [
    ... ('foo', [('1', 1.0)]),
    ... ('bar', [('1', 2/3)]),
    ... ('baz', [('1', 1/3)]),
    ... ]
    True
    """
    def __init__(self):
        self.calls = {}

    def get_freq(self, call, sym):
        return self.calls.get(call, EmptyCounter)[sym]

    def inc_freq(self, call, sym):
        if call not in self.calls:
            counter = Counter()
            self.calls[call] = counter
        else:
            counter = self.calls[call]
        counter[sym] += 1

    def update_max_freq(self):
        # Update max frequencies
        self.max_freqs = Counter()
        for call, freqs in self.calls.items():
            for sym, freq in freqs.items():
                self.max_freqs[sym] = max(self.max_freqs[sym], freq)

    def get_weight(self, call, sym):
        freq = self.get_freq(call, sym)
        max_freq = self.max_freqs[sym]
        return freq/max_freq

    def build(self):
        calls = {}
        for call, syms in self.calls.items():
            group = ((sym, freq / self.max_freqs[sym]) \
                for sym, freq in syms.items())
            yield call, group

class StateAnomalyDB:
    """
    >>> db = StateAnomalyDB(1)
    >>> len(db.state_vars)
    1
    >>> db.add_anomaly('foo', 0, '1')
    >>> db.add_anomaly('foo', 0, '1')
    >>> db.add_anomaly('foo', 0, '1')
    >>> db.state_vars[0].calls
    {'foo': Counter({'1': 3})}
    >>> db.update_max()
    """
    def __init__(self, state_count) -> None:
        state_vars = list((StateVar() for _ in range(state_count)))
        self.state_vars = state_vars

    def add_anomaly(self, call, idx, sym):
        self.state_vars[idx].inc_freq(call, sym)

    def update_max(self):
        for state_var in self.state_vars:
            state_var.update_max_freq()

    def build(self):
        for var_st in self.state_vars:
            yield var_st.build()

    def group_by_call(self):
        """
        >>> db = StateAnomalyDB(1)
        >>> len(db.state_vars)
        1
        >>> db.add_anomaly('foo', 0, '1')
        >>> db.add_anomaly('foo', 0, '1')
        >>> db.add_anomaly('foo', 0, '1')
        >>> db.state_vars[0].calls
        {'foo': Counter({'1': 3})}
        >>> db.update_max()
        >>> db.group_by_call()
        {'foo': [{'1': 1.0}]}
        """
        result = {}
        for idx, var_st in enumerate(self.build()):
            for call, syms in var_st:
                # Get all the values associated with a call
                if call not in result:
                    per_call = list({} for _ in range(len(self.state_vars)))
                    result[call] = per_call
                else:
                    per_call = result[call]
                call_entries = per_call[idx]
                for sym, weight in syms:
                    # Each call should have a 
                    call_entries[sym] = weight
        return result

def get_anomalous_states(call, threshold):
    for idx, st in enumerate(call.states):
        if st.normalized_prob < threshold and st.get_max()[1] >= threshold:
            yield (idx, st)

def load_state_anomaly(seqs, state_count, threshold, accept_state=lambda x:True):
    db = StateAnomalyDB(state_count)
    visited = set()
    for seq in seqs:
        for evt, call in zip(seq, seq.get_state_probs()):
            for idx, st in get_anomalous_states(call, threshold):
                if not accept_state(st):
                    continue
                cid = (evt.location, call.name, idx)
                if cid in visited:
                    continue
                visited.add(cid)
                db.add_anomaly(call.name, idx, st.name)
    db.update_max()
    return db.group_by_call()
