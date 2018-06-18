from operator import *
from typing import *

import numpy as np
from enum import Enum, auto
from functools import partial, lru_cache
from salento.models.low_level_evidences.infer import VectorMapping

def partition_vocab(terms:List[str], sentinel:str='STOP') -> \
        Tuple[
            List[int],
            Dict[int, List[int]]
        ]:
    """
    Given a list of the vocabolary, returns the index of each call name
    and of each state.

        >>> partition_vocab(['foo', 'bar', '0#asd', 'STOP', '0#bsd', '3#'])
        ([0, 1, 3], {0: [2, 3, 4], 3: [3, 5]})
    """
    states:Dict[int, List[int]] = {}
    calls:List[int] = []
    sentinel_idx = -1
    for idx, term in enumerate(terms):
        if "#" in term:
            try:
                num = int(term.split("#", 1)[0])
                if num >= 0:
                    state = states.get(num)
                    if state is None:
                        state = []
                        states[num] = state
                    state.append(idx)
                    continue
            except ValueError:
                pass
        elif term == sentinel:
            sentinel_idx = idx
        # if we reach this, then we are in the presence of a call
        calls.append(idx)
    if sentinel_idx >= 0:
        for (k, v) in states.items():
            v.append(sentinel_idx)
            v.sort()
    return calls, states

def decode_state(x:str, sentinel='STOP') -> str:
    if x == sentinel:
        return x
    try:
        return x.split("#", 1)[1]
    except IndexError:
        raise ValueError("Expecting an encoded state, but got: %r" % x)

def encode_state(state:int, data:Any) -> str:
    return str(state) + "#" + str(data)

class EagerRestriction:
    """
    Check loading of an eager restriction object:

        >>> vocab = ['foo', 'bar', '0#asd', '0#bsd', '3#']
        >>> calls = partition_vocab(vocab)[0]
        >>> res = EagerRestriction(np.array(calls), np.array(vocab))
        >>> list(res.indices)
        [0, 1]
        >>> list(res.id_to_term)
        ['foo', 'bar']
        >>> res.term_to_id
        {'foo': 0, 'bar': 1}

    Check adapting:

        >>> vm = VectorMapping(data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), id_to_term=vocab,
        ... term_to_id={'foo': 0, 'bar': 1, '0#asd':2, '0#bsd':3, '3#':4})
        >>> vm2 = res(vm)
        >>> list(vm2.data)
        [0.1, 0.2]
        >>> vm2.id_to_term is res.id_to_term
        True
        >>> vm2.term_to_id is res.term_to_id
        True
        >>> dict(vm2.items())
        {'foo': 0.1, 'bar': 0.2}
        >>> list(vm2.keys())
        ['foo', 'bar']
        >>> vm2['bar']
        0.2

    Check term translation:

        >>> vocab = ['foo', 'bar', '0#asd', '0#bsd', '3#']
        >>> states = partition_vocab(vocab)[1][0]
        >>> arr_decode_state = np.vectorize(decode_state)
        >>> res = EagerRestriction(np.array(states), np.array(vocab), arr_decode_state)
        >>> res.term_to_id
        {'asd': 0, 'bsd': 1}
        >>> vm2 = res(vm)
        >>> dict(vm2.items())
        {'asd': 0.3, 'bsd': 0.4}
        >>> list(vm2.keys())
        ['asd', 'bsd']
        >>> vm2['asd']
        0.3
    """
    def __init__(self, indices:np.ndarray, terms:np.ndarray, vocab_translator:Optional[Callable[[np.ndarray], np.ndarray]]=None ) -> None:
        if not isinstance(terms, np.ndarray):
            raise ValueError("Terms must be an array.")
        if not isinstance(indices, np.ndarray):
            raise ValueError("Indices must be an array.")
        self.indices = indices
        self.id_to_term = terms[self.indices]
        if vocab_translator is not None:
            self.id_to_term = vocab_translator(self.id_to_term)
        self.term_to_id = dict(zip(self.id_to_term, range(len(self.id_to_term))))

    def __call__(self, data:VectorMapping) -> VectorMapping:
        return VectorMapping(
            data=data.data[self.indices],
            id_to_term=self.id_to_term,
            term_to_id=self.term_to_id,
        )

def make_zero(data, sentinel='STOP'):
    idx = data.term_to_id[sentinel]
    return VectorMapping(
        data=np.array([data.data[idx]]),
        id_to_term=[sentinel],
        term_to_id={sentinel: 0},
    )


TermTranslator = Callable[[str],str]


class LazyRestriction:
    """
    Restricts the domain of a map given a predicate.

    Check loading of an eager restriction object:

        >>> vocab = ['foo', 'bar', '0#asd', '0#bsd', '3#']
        >>> calls = np.array([0, 1])
        >>> vm = VectorMapping(data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), id_to_term=vocab,
        ... term_to_id={'foo': 0, 'bar': 1, '0#asd':2, '0#bsd':3, '3#':4})

        >>> res = LazyRestriction(calls, vm)
        >>> res.indices is calls
        True
        >>> res.index_set
        {0, 1}
        >>> dict(res.items())
        {'foo': 0.1, 'bar': 0.2}
        >>> list(res.keys())
        ['foo', 'bar']
        >>> res['bar']
        0.2
        >>> res.get_max()
        ('bar', 0.2)

    Check term translation:

        >>> vocab = ['foo', 'bar', '0#asd', '0#bsd', '3#']
        >>> states = partition_vocab(vocab)[1][0]
        >>> encoder = partial(encode_state, 0)
        >>> vm2 = LazyRestriction(np.array(states), vm, term_encoder=encoder, term_decoder=decode_state)
        >>> dict(vm2.items())
        {'asd': 0.3, 'bsd': 0.4}
        >>> list(vm2.keys())
        ['asd', 'bsd']
        >>> vm2['asd']
        0.3
        >>> vm2.get_max()
        ('bsd', 0.4)

    """

    def __init__(self, indices:np.ndarray,
            mapping:VectorMapping,
            term_encoder:Optional[TermTranslator]=None,
            term_decoder:Optional[TermTranslator]=None) -> None:
        self.indices = indices
        self.index_set = set(indices)
        self.mapping = mapping
        id_to_term:Callable[[int],str] = self.mapping.id_to_term.__getitem__
        if term_decoder is not None:
            self.id_to_term:Callable[[int],str] = lambda x: term_decoder(id_to_term(x))
        else:
            self.id_to_term:Callable[[int],str] = id_to_term

        term_to_id:Callable[[str],int] = self.mapping.term_to_id.__getitem__
        if term_encoder is not None:
            self.term_to_id = lambda x: term_to_id(term_encoder(x))
        else:
            self.term_to_id = term_to_id

    def __getitem__(self, key:str) -> float:
        idx = cast(Callable[[str],int], self.term_to_id)(key)
        if idx in self.index_set:
            return self.mapping.data[idx]
        else:
            raise KeyError(key)

    def items(self) -> Iterator[Tuple[str, float]]:
        id_to_term:Callable[[int],str] = self.id_to_term
        data = self.mapping.data
        for idx in self.indices:
            yield (id_to_term(idx), data[idx])

    def keys(self) -> Iterator[str]:
        id_to_term:Callable[[int],str] = self.id_to_term
        for idx in self.indices:
            yield id_to_term(idx)

    def get_max(self) -> Tuple[str,float]:
        data = self.mapping.data
        idx, amount = max(((idx,data[idx]) for idx in self.indices), key=itemgetter(1))
        term = self.id_to_term(idx)
        return (term, amount)

class RestrictionStrategy(Enum):
    # Currently the default implementation: partitions the vocabolary and gets
    # the indices that represent one partition (say, the subset of all calls).
    # Given a distribution prob, we use Numpy's fast implementation to
    # restrict the input array to a new one (a copy), yielding the new dist
    # probability.
    EAGER = auto()
    # LAZY performs the partition upon loading but does create a new array.
    # Given a distribution prob, a set of keys/indices is pre-computed
    # whenever we query an element convert one key to the indice in the
    # underlying array.
    LAZY = auto()
    # Nothing is pre-computed (best memory efficiency); computes the set of
    # keys of interest on-demand. This is the slowest algorithm.
    LAZY2 = auto()

DistAdapter = Callable[[VectorMapping],VectorMapping]

class TermCodec:
    term_encoder: Optional[TermTranslator]
    term_decoder: Optional[TermTranslator]
    vocab_translator: Optional[Callable[[np.ndarray],np.ndarray]]

    def __init__(self,
            term_encoder:Optional[TermTranslator],
            term_decoder:Optional[TermTranslator]) -> None:
        self.term_encoder = term_encoder
        self.term_decoder = term_decoder
        if term_encoder is not None:
            self.vocab_translator = np.vectorize(term_decoder)
        else:
            self.vocab_translator = None

def restrict_vector_mapping(indices:np.ndarray, vocab:np.ndarray, term_codec:TermCodec=TermCodec(None, None), strategy:RestrictionStrategy=RestrictionStrategy.EAGER) \
        -> DistAdapter:
    """
        >>> vocab = ['foo', 'bar', '0#asd', '0#bsd', '3#']
        >>> states = partition_vocab(vocab)[1][0]
        >>> codec = TermCodec(partial(encode_state, 0), decode_state)
        >>> res = restrict_vector_mapping(np.array(states), np.array(vocab), codec, strategy=RestrictionStrategy.EAGER)
        >>> vm = VectorMapping(data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), id_to_term=vocab,
        ... term_to_id={'foo': 0, 'bar': 1, '0#asd':2, '0#bsd':3, '3#':4})

        >>> vm2 = res(vm)
        >>> dict(vm2.items())
        {'asd': 0.3, 'bsd': 0.4}
        >>> list(vm2.keys())
        ['asd', 'bsd']
        >>> vm2['asd']
        0.3
        >>> vm2.get_max()
        ('bsd', 0.4)

        >>> res = restrict_vector_mapping(np.array(states), np.array(vocab), codec, strategy=RestrictionStrategy.LAZY)
        >>> vm2 = res(vm)
        >>> dict(vm2.items())
        {'asd': 0.3, 'bsd': 0.4}
        >>> list(vm2.keys())
        ['asd', 'bsd']
        >>> vm2['asd']
        0.3
        >>> vm2.get_max()
        ('bsd', 0.4)
    """
    if strategy == RestrictionStrategy.EAGER:
        return EagerRestriction(indices, vocab,
            vocab_translator=term_codec.vocab_translator)
    elif strategy == RestrictionStrategy.LAZY:
        return cast(DistAdapter, partial(LazyRestriction, indices,
            term_encoder=term_codec.term_encoder,
            term_decoder=term_codec.term_decoder))
    else:
        raise ValueError("Unknown strategy: " + repr(strategy))

class DistFilter:
    def filter_call_names(self, term_dist:VectorMapping) -> VectorMapping:
        pass

    def filter_state(self, idx:int, term_dist:VectorMapping) -> VectorMapping:
        pass

class StaticDistFilter(DistFilter):
    """
        >>> vocab = np.array(['foo', 'bar', '0#asd', '0#bsd', '3#'])
        >>> term_res = StaticDistFilter(vocab)
        >>> list(term_res.states.keys())
        [0, 3]
        >>> vm = VectorMapping(data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), id_to_term=vocab,
        ... term_to_id={'foo': 0, 'bar': 1, '0#asd':2, '0#bsd':3, '3#':4})

        >>> vm2 = term_res.filter_state(0, vm)
        >>> dict(vm2.items())
        {'asd': 0.3, 'bsd': 0.4}
        >>> list(vm2.keys())
        ['asd', 'bsd']
        >>> vm2['asd']
        0.3

        >>> vm2 = term_res.filter_call_names(vm)
        >>> dict(vm2.items())
        {'foo': 0.1, 'bar': 0.2}
        >>> list(vm2.keys())
        ['foo', 'bar']
        >>> vm2['foo']
        0.1
    """

    def __init__(self, vocab:np.ndarray, strategy:RestrictionStrategy=RestrictionStrategy.EAGER) -> None:
        calls, states = partition_vocab(vocab)
        # Initialize the various mappings:
        self.calls:DistAdapter = restrict_vector_mapping(np.array(calls), vocab, strategy=strategy)
        self.states:Dict[int,DistAdapter] = {}
        for k, v in states.items():
            term_codec = TermCodec(
                term_encoder=partial(encode_state, k),
                term_decoder=decode_state
            )
            self.states[k] = restrict_vector_mapping(np.array(v), vocab,
                term_codec=term_codec,
                strategy=strategy)

    def filter_call_names(self, term_dist:VectorMapping) -> VectorMapping:
        return self.calls(term_dist)

    def filter_state(self, idx:int, term_dist:VectorMapping) -> VectorMapping:
        adapter = self.states.get(idx)
        if adapter is None:
            adapter = make_zero
        return adapter(term_dist)

##########

def filter_call_names(key):
    return "#" not in key

def filter_state(idx):
    expected_key = str(idx) + "#"
    def do_filter_state(key):
        return key == 'STOP' or key.startswith(expected_key)
    return do_filter_state

class FilteredMap:
    def __init__(self, data, predicate, term_encoder=lambda x:x, term_decoder=lambda x:x):
        assert data is not None
        assert predicate is not None
        self.data = data
        self.predicate = predicate
        self.encode_term = term_encoder
        self.decode_term = term_decoder

    def __getitem__(self, key):
        return self.data[self.encode_term(key)]

    def items(self):
        pred = self.predicate
        dec = self.decode_term
        terms = filter(lambda x: pred(x[0]), self.data.items())
        return ((dec(k), v) for k, v in terms)

    def keys(self):
        return filter(self.predicate, map(self.decode_term, self.data.keys()))

    def get_max(self):
        return max(self.items(), key=itemgetter(1))

class DynamicDistFilter(DistFilter):
    def filter_call_names(self, term_dist):
        return FilteredMap(term_dist, filter_call_names)

    def filter_state(self, idx, term_dist):
        return FilteredMap(term_dist, filter_state(idx),
            term_encoder=partial(encode_state, idx),
            term_decoder=decode_state
        )

DYNAMIC_DIST_FILTER = DynamicDistFilter()

class TermDistNorm:
    dist:VectorMapping

    def __init__(self, name:str, dist:VectorMapping) -> None:
        self.name:str = name
        self.prob:float = dist[name]
        self.dist:VectorMapping = dist

    @lru_cache(maxsize=None)
    def get_max(self) -> Tuple[str,float]:
        return self.dist.get_max()

    @property
    def normalized_prob(self) -> float:
        return self.prob / self.get_max()[1]

class CallDistNorm(TermDistNorm):
    def __init__(self, name:str, dist:VectorMapping) -> None:
        super().__init__(name, dist)
        self.states:List[TermDistNorm] = []

    def __iter__(self) -> Iterator[TermDistNorm]:
        yield self
        yield from self.states

    def __len__(self) -> int:
        return 1 + len(self.states)

Row = List[Tuple[str, VectorMapping]]

def adapt_state_distribution(dist_filter:DistFilter, row:Row) -> CallDistNorm:
    call_name, call_dist = row[0]
    call = CallDistNorm(
        call_name,
        dist_filter.filter_call_names(call_dist)
    )
    for idx, (key, dist) in enumerate(row[1:]):
        new_dist = dist_filter.filter_state(idx, dist)
        key = decode_state(key)
        call.states.append(TermDistNorm(key, new_dist))
    return call

def create_adapter(vocab:np.ndarray, strategy=RestrictionStrategy.EAGER) -> Callable[[Row], CallDistNorm]:
    """
    Creates a state-distribution adapter.
    Takes a vocabolary (a numpy array) and creates a state distribution adapter.

    The usage is as follows. Given an instance of `Aggregator` take the vocabolary
    of the trained model and use it to build a states-distribution adapter.

        adapter = statedist.create_adapter(agg.model.model.config.decoder.chars)

    Then use the adpater to partition the results of `distribution_state_iter`:

        for row in app.distribution_state_iter(spec, call_seq):
            call:CallDistNorm = app.dist_adapter(row)
            # A call has a name and a distribution (see DistTermNorm):
            # call.name is the name of the call
            # call.prob is the raw probability of that call
            # call.dist is the distribution probability at that call
            # call.normalized_prob is `call.prob/call.get_max()[1]`
            # Additionally, a call has call.states, which is a list of all
            # states associated with the given call, each of which is call is a
            # DistTermNorm
            # Finally, a call also has an iterator that returns self and all
            # the states of this call.

    Here is a self-contained example. First, we prepare a vocabolary and an
    adapter object (for the sake of testing  we are using the LAZY2
    implementation, this is the slowest implementation, the default
    implementation is the fastest):

        >>> vocab = np.array(['foo', 'bar', '0#asd', '0#bsd', '3#'])
        >>> adapter = create_adapter(vocab, strategy=RestrictionStrategy.LAZY2)

    Next we build a row object:

        >>> vm = VectorMapping(data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), id_to_term=vocab,
        ... term_to_id={'foo': 0, 'bar': 1, '0#asd':2, '0#bsd':3, '3#':4})
        >>> row = [('foo', vm), ('0#asd', vm)]

    Finally, we take a row object and convert it to a DistTerm object:

        >>> call = adapter(row)

    And now we inspect it:

        >>> len(call.states)
        1
        >>> call.name
        'foo'
        >>> call.prob
        0.1
        >>> dict(call.dist.items())
        {'foo': 0.1, 'bar': 0.2}

    We can also inspect its states:

        >>> s = call.states[0]
        >>> s.name
        'asd'
        >>> s.prob
        0.3
        >>> dict(s.dist.items())
        {'asd': 0.3, 'bsd': 0.4}

    """
    if strategy == RestrictionStrategy.LAZY2:
        return partial(adapt_state_distribution, DYNAMIC_DIST_FILTER)
    
    return partial(adapt_state_distribution, StaticDistFilter(vocab, strategy=strategy))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # Benchmark our three implementations

    # Each run takes a state distribution of this size:
    count = 1_000_000
    # There is only one state, which uses this percentage of the vocabulary:
    state_count_ratio = 0.005
    # The benchmark computes the maximum probabilty of the state distribution
    # these many times:
    loop = 5

    # The benchmark is below
    import time
    import random
    def _rand_vocab(count):
        states = int(count * state_count_ratio)
        vocab = []
        for k in range(count - states):
            vocab.append("call" + str(k))
        for k in range(count - states, count):
            vocab.append("0#state" + str(k + 1 - count))
        random.shuffle(vocab)
        return np.array(vocab), dict((v,k) for k, v in enumerate(vocab))

    def _rand_vector_mapping(count, vocab, term_to_id):
        return VectorMapping(data=np.random.random_sample((count,)), id_to_term=vocab, term_to_id=term_to_id)

    vocab, term_to_id = _rand_vocab(count)
    eager_a = create_adapter(vocab, strategy=RestrictionStrategy.EAGER)
    lazy_a = create_adapter(vocab, strategy=RestrictionStrategy.LAZY)
    lazy2_a = create_adapter(vocab, strategy=RestrictionStrategy.LAZY2)

    vm = _rand_vector_mapping(count, vocab, term_to_id)
    row = [('call0', vm), ('0#state0', vm)]

    results = []

    start = time.time()
    for _ in range(loop):
        for x in eager_a(row):
            x.get_max()
    end = time.time()
    results.append(("eager", end - start))

    start = time.time()
    for _ in range(loop):
        for x in lazy_a(row):
            x.get_max()
    end = time.time()
    results.append(("lazy", end - start))

    start = time.time()
    for _ in range(loop):
        for x in lazy2_a(row):
            x.get_max()
    end = time.time()
    results.append(("lazy2", end - start))

    winner, winner_time = min(results, key=itemgetter(1))
    print("vocab size:", count)
    print("state-0 size:", int(count * state_count_ratio))
    print("runs:", loop)
    for k, v in results:
        extra = ("({}x speedup)".format(int(v/winner_time)) if k != winner else "")
        print(f"{k}:\t{v:.2f}s {extra}")
    print("winner:", winner)

