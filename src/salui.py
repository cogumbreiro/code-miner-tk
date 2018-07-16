from typing import *
import numpy as np
import weakref

import sal
import statedist
from common import memoize, as_list

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
