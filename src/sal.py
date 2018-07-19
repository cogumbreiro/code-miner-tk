from operator import *
import collections
import itertools
from common import from_slice

END_MARKER = 'STOP'

def get_sequences(pkg):
    return pkg['data']

def get_package_name(pkg):
    return pkg['name']

def get_calls(seq):
    return seq['sequence']

def get_packages(doc):
    if "packages" in doc:
        return doc['packages']
    else:
        return (doc,)

def get_call_name(call):
    return call['call']

def get_call_location(call):
    return call['call']

def eq_iter(l, r):
    if len(l) != len(r):
        return False

    for x,y in zip(l, r):
        if x != y: return False
    return True

class Dataset:
    """
    >>> d = Dataset(js={'packages': [{'name': 'foo', 'data': []}]})
    >>> list(d)
    [Package([], name='foo', pid=0)]
    >>> Dataset([Package([], name='foo', pid=0)])
    Dataset([Package([], name='foo', pid=0)])
    """
    def __init__(self, packages=None, js=None, adapt_from_package=False):
        if js is None:
            self.js = {'packages': []}

        else:
            if adapt_from_package and ("packages" not in js and "data" in js):
                js = {"packages": [js]}

            self.js = js

        if packages is not None:
            self.js['packages'] = list(map(lambda x:x.js, packages))

    def make_package(self, js, pid):
        return Package(js=js, pid=pid)

    def __iter__(self):
        for pid, pkg in enumerate(self.js['packages']):
            yield self.make_package(js=pkg, pid=pid)

    def __len__(self):
        return len(self.js['packages'])

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = map(attrgetter("js"), value)

        data = self.js["packages"].__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return from_slice(key, self)
        else:
            return self.make_package(js=self.js['packages'][key], pid=key)

    __eq__ = eq_iter

    @property
    def count(self):
        return len(self)

    def __repr__(self):
        return 'Dataset(%r)' % list(self)

    def translate_calls(self, alias):
        """
        We can supply a map of aliases; the terms are replaced before filtering.

            >>> seq1 = Sequence([Call('baz'), Call('X'), Call('Y'), Call('bar')])
            >>> pkg = Package([seq1], name='p')
            >>> ds = Dataset([pkg])
            >>> ds.translate_calls({'baz': 'foo', 'bar': 'ZZZ'})
            >>> len(ds)
            1
            >>> len(ds[0])
            1
            >>> list(ds[0][0].terms)
            ['foo', 'X', 'Y', 'ZZZ']
        """
        for pkg in get_packages(doc=self.js):
            for seq in get_sequences(pkg=pkg):
                for term in get_calls(seq=seq):
                    call = term['call']
                    if call in alias:
                        term['call'] = alias[call]

    def apply_call_filter(self, call_filter, branch_tokens):
        do_filter = make_filter_branch(call_filter, branch_tokens=branch_tokens)
        for pkg in get_packages(doc=self.js):
            for seq in get_sequences(pkg=pkg):
                seq['sequence'][:] = do_filter(seq)


    def filter_vocabs(self, vocabs, branch_tokens=set(['$BRANCH']), call_filter=None):
        """
        There's a notion of a branch token: when a non-branch token is
        removed (because it is a stop word or because it is not in the vocabs),
        all succeeding branch tokens are removed. In the following example we have
        two branch tokens that are removed because 'foo' is removed.

            >>> seq1 = Sequence([Call('foo'), Call('X'), Call('Y'), Call('bar')])
            >>> pkg = Package([seq1], name='p', pid=0)
            >>> ds = Dataset([pkg])
            >>> ds.filter_vocabs(['bar'], branch_tokens=('X','Y'))
            >>> len(ds), len(ds[0])
            (1, 1)
            >>> list(ds[0][0].terms)
            ['bar']
        """
        f = make_filter_vocabs(vocabs)
        if call_filter is not None:
            f = call_filter(f)
        self.apply_call_filter(f, branch_tokens)

    def filter_stopwords(self, stopwords, branch_tokens=set(['$BRANCH']), call_filter=None):
        """
        There's a notion of a branch token: when a non-branch token is
        removed (because it is a stop word or because it is not in the vocabs),
        all succeeding branch tokens are removed. In the following example we have
        two branch tokens that are removed because 'foo' is removed.

            >>> seq1 = Sequence([Call('foo'), Call('X'), Call('Y'), Call('bar')])
            >>> pkg = Package([seq1], name='p', pid=0)
            >>> ds = Dataset([pkg])
            >>> ds.filter_stopwords(['foo'], branch_tokens=('X','Y'))
            >>> len(ds), len(ds[0])
            (1, 1)
            >>> list(ds[0][0].terms)
            ['bar']
        """
        f = make_filter_stopwords(stopwords)
        if call_filter is not None:
            f = call_filter(f)
        self.apply_call_filter(f, branch_tokens)

    def flatten_sequences(self, inline=True):
        """
        Sequences of calls are breaken into single calls:

            >>> s = Sequence([Call('hey', location='foo'), Call('there')])
            >>> ds = Dataset([Package([s])])
            >>> ds.flatten_sequences()
            >>> len(ds) == 1
            True
            >>> len(ds[0]) == 2
            True
            >>> list(map(list, ds[0])) == [[Call('hey', location='foo', cid=0)], [Call('there', cid=0)]]
            True

        States are serialized as part of the call-sequence:

            >>> s = Sequence([Call('hey', location='loc', states=[1,"foo"])])
            >>> ds = Dataset([Package([s])])
            >>> ds.flatten_sequences()
            >>> len(ds) == 1
            True
            >>> len(ds[0]) == 1
            True
            >>> pkg = list(map(list, ds[0]))
            >>> seq, = pkg # There is only one sequence
            >>> seq == [
            ...    Call('hey', location='loc', cid=0),
            ...    Call('1', location='loc', cid=1),
            ...    Call('foo', location='loc', cid=2)
            ... ]
            True

        A sequence is broken into multiple sequences:

            >>> s = Sequence([Call('foo', states=[1,2]), Call('bar', states=["s"])])
            >>> ds = Dataset([Package([s])])
            >>> ds.flatten_sequences()
            >>> len(ds) == 1
            True
            >>> pkg = ds[0]
            >>> len(pkg)
            2
            >>> seqs = list(map(lambda x:list(x.terms), ds[0]))
            >>> seqs
            [['foo', '1', '2'], ['bar', 's']]

        A sequence is broken into multiple sequences:

            >>> s = Sequence([Call('foo', states=[1,2]), Call('bar', states=["s"])])
            >>> ds = Dataset([Package([s])])
            >>> ds.flatten_sequences(inline=False)
            >>> len(ds) == 1
            True
            >>> pkg = ds[0]
            >>> list(pkg) == [
            ... Sequence([Call('foo', states=[1,2])]),
            ... Sequence([Call('bar', states=["s"])]),
            ... ]
            True
        """
        for pkg in get_packages(doc=self.js):
            # Retreive the sequences
            seqs = get_sequences(pkg=pkg)
            # Cleanup the old sequences
            new_seqs = []
            pkg['data'] = new_seqs

            for seq in seqs:

                for call in get_calls(seq=seq):
                    if inline:
                        states = call['states'] # cache states
                        loc = call['location']  # cache location
                        call['states'] = []     # remove old states
                        new_calls = [call]      # the sequence starts with the call
                        for s in states:
                            new_calls.append({'location': loc, 'states':[], 'call': "{}".format(s)})
                        # append a new sequnce per call
                        new_seqs.append({'sequence': new_calls})
                    else:
                        new_seqs.append({'sequence': [call]})


    def filter_sequences(self, min_length=2):
        do_filter = lambda s: len(s['sequence']) >= min_length
        for pkg in get_packages(doc=self.js):
            pkg['data'][:] = filter(do_filter, get_sequences(pkg=pkg))




class Package:
    '''
    >>> js_c = {'call': 'foo', 'location': 'bar', 'states': [1]}
    >>> js_s = {'sequence': [js_c]}
    >>> s = Sequence(js=js_s)
    >>> js = {'name': 'baz', 'data': [js_s]}
    >>> pkg = Package(js=js)
    >>> pkg
    Package([Sequence([Call(cid=0, call='foo', location='bar', states=[1])], sid=0)], name='baz', pid=-1)
    >>> s = Sequence(js=js_s, sid=0)
    >>> pkg.name
    'baz'
    >>> len(pkg)
    1
    >>> pkg[0] == s
    True
    '''

    def __init__(self, sequences=None, name=None, pid=-1, js=None):
        if js is None:
            self.js = {'data': [], 'name': ''}
        else:
            self.js = js
        if name is not None:
            self.name = name
        if sequences is not None:
            self.js['data'] = list(map(lambda x:x.js, sequences))
        self.pid = pid

    def group_by_last_location(self):
        by_last = attrgetter("last_location")
        # remove empty sequences
        elems = sorted(filter(lambda seq: len(seq) > 0, self), key=by_last)
        # Group by last location
        return itertools.groupby(elems, by_last)

    @property
    def count(self):
        return len(self)

    @property
    def name(self):
        return self.js['name']

    @name.setter
    def name(self, value):
        self.js['name'] = value

    def __eq__(self, other):
        return self.name == other.name and eq_iter(self, other)

    def make_sequence(self, js, sid):
        return Sequence(sid=sid, js=js)

    def __iter__(self):
        for sid, seq in enumerate(get_sequences(pkg=self.js)):
            yield self.make_sequence(js=seq, sid=sid)

    def __len__(self):
        return len(self.js['data'])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return from_slice(key, self)
        else:
            return self.make_sequence(sid=key, js=get_sequences(self.js)[key])

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = map(attrgetter("js"), value)

        data = self.js['data'].__setitem__(key, value)

    def __repr__(self):
        return 'Package(%r, name=%r, pid=%r)' % (list(iter(self)), self.name, self.pid)


def match(src, name):
    src = src.lower()
    name = name.lower()
    if name.startswith('^') and name.endswith('$'):
        name = name[1:-1]
        return src == name
    if name.endswith('$'):
        name = name[:-1]
        return src.endswith(name)
    if name.endswith('^'):
        name = name[1:]
        return src.startswith(name)
    return name in src

def call_name(call):
    return "{}:{}".format(call['call'],call['location'])

class make_filter_branch:
    """
    Given a sequence, returns all the terms that match a given
    call-predicate, while respecting branches.

    When a non-branch token is removed, all succeeding branch tokens are
    removed. In the following example we have two branch tokens that are
    removed because 'foo' is removed.

        >>> seq = Sequence([Call('foo'), Call('X'), Call('Y'), Call('bar')])
        >>> f = make_filter_branch(lambda x:x['call'] == 'bar', branch_tokens=('X','Y'))
        >>> list(map(lambda x:x['call'], f(seq.js)))
        ['bar']

    Next, we show that the call-predicate is not invoked for branch-terms:

        >>> seq = Sequence([Call('foo'), Call('X'), Call('Y'), Call('bar')])
        >>> f = make_filter_branch(lambda x:x['call'] == 'foo', branch_tokens=('X','Y'))
        >>> list(map(lambda x:x['call'], f(seq.js)))
        ['foo', 'X', 'Y']

    """
    def __init__(self, predicate, branch_tokens=set(['$BRANCH'])):
        self.predicate = predicate
        self.branch_tokens = branch_tokens

    def __call__(self, seq):
        to_remove = False
        for term in get_calls(seq=seq):
            name = term['call']
            # This branch is needed because if we remove a term, we must remove
            # the consecutive $BRANCH tokens if they exist
            if to_remove:
                if name in self.branch_tokens:
                    continue
                to_remove = False

            if name in self.branch_tokens or self.predicate(term):
                to_remove = False
                yield term
            else:
                to_remove = True


class make_filter_on_reject:
    """
    Logs rejected terms.
    """
    def __init__(self, predicate, on_reject):
        self.predicate = predicate
        self.on_reject = on_reject

    def __call__(self, term):
        result = self.predicate(term)
        if not result:
            self.on_reject(term)
        return result

def make_filter_combine(first, second):
    return lambda x: first(x) and second(x)

def make_filter_vocabs(vocabs):
    """

    We can vocabs to limit the accepted terms, in this case by removing
    the call 'bar' (note that we are not filtering out based on minimum length):

        >>> f = make_filter_vocabs(['foo', 'baz'])
        >>> f(Call('foo').js)
        True
        >>> f(Call('box').js)
        False
        >>> f(Call('baz').js)
        True

    """
    return lambda x: x['call'] in vocabs

def make_filter_stopwords(stopwords):
    """
    We can use stop words to eliminate calls, in this case by removing
    the call 'foo' we actually remove the first sequence (as it falls below
    the acceptable minimum length):

        >>> f = make_filter_stopwords(stopwords=['bar'])
        >>> f(Call('foo').js)
        True
        >>> f(Call('bar').js)
        False
    """
    return lambda x: x['call'] not in stopwords



class Sequence:
    """
    Given some sequence object which we build from a JSON object:

        >>> js_c1 = {'call': 'foo', 'location': 'bar', 'states': [1]}
        >>> js_c2 = {'call': 'foo2', 'location': 'bar2', 'states': []}
        >>> js = {'sequence': [js_c1, js_c2]}
        >>> c = Call(cid=0, call='foo', location='bar', states=[1])
        >>> x = Sequence(js=js)
        >>> x
        Sequence([Call(cid=0, call='foo', location='bar', states=[1]), Call(cid=1, call='foo2', location='bar2', states=[])], sid=-1)

    Len also works as expected:

        >>> len(x)
        2

    Get-item also works:

        >>> x[0] == c
        True

    You can also retreive the last location of a sequence:

        >>> x.last_location
        'bar2'

    Slice-reading work as expected:

        >>> x[:]
        [Call(cid=0, call='foo', location='bar', states=[1]), Call(cid=1, call='foo2', location='bar2', states=[])]

    Slice-writing work as expected:

        >>> x[0:1] = [Call('one'), Call('two')]
        >>> list(x.terms)
        ['one', 'two', 'foo2']

    A list of calls.

    Given a sequence of calls:

        >>> c = Call(call='foo', location='bar', states=[1], cid=0)
        >>> x = Sequence([c])
        >>> x
        Sequence([Call(cid=0, call='foo', location='bar', states=[1])], sid=-1)

    You can retreive the last location of a sequence:

        >>> x.last_location
        'bar'

    The length works as expected:

        >>> len(x)
        1

    Get-item also works:
        >>> x[0] == c
        True

    """
    def __init__(self, sequence=None, sid=-1, js=None):
        if js is None:
            self.js = {'sequence': []}
        else:
            self.js = js

        if sequence is not None:
            self.js['sequence'] = list(map(lambda x:x.js, sequence))

        self.sid = sid

    __eq__ = eq_iter

    def __hash__(self):
        return hash(tuple(self))

    @property
    def count(self):
        return len(self)

    @property
    def last_location(self):
        return self[-1].location

    def __repr__(self):
        return 'Sequence(%r, sid=%r)' % (list(iter(self)), self.sid)

    def as_string(self, include_state=True, include_location=True,
            seq_sep="\n", state_sep=",", field_sep=":"):
        """
        Converts a sequence to a string representation.
        """
        return seq_sep.join(map(
            lambda x: x.as_string(
                include_state=include_state,
                include_location=include_location,
                state_sep=state_sep,
                field_sep=field_sep,
            ),
            self
        ))

    def matches_at(self, name, idx, key):
        return match(key(self[idx]), name)

    def matches_any(self, name, key):
        name = name.lower()
        for call in self:
            if match(key(call), name):
                return True
        return False

    @property
    def terms(self):
        return map(attrgetter("call"), self)

    def make_call(self, js, cid):
        return Call(js=js, cid=cid)

    def __iter__(self):
        for cid, call in enumerate(get_calls(self.js)):
            yield self.make_call(cid=cid, js=call)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            value = map(attrgetter("js"), value)

        self.js['sequence'].__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return from_slice(key, self)
        else:
            return self.make_call(cid=key, js=get_calls(self.js)[key])

    def __delitem__(self, key):
        del self.js[key]

    def __len__(self):
        return len(get_calls(self.js))




class Call:
    """
    >>> x = Call(js={'call': 'foo', 'location': 'bar', 'states': [1]})
    >>> x.cid
    -1
    >>> x.call
    'foo'
    >>> x.location
    'bar'
    >>> x.states
    [1]
    >>> x
    Call(cid=-1, call='foo', location='bar', states=[1])
    >>> c = Call(call='foo', location='bar', states=[1])
    >>> c.call
    'foo'
    >>> c.location
    'bar'
    >>> c.states
    [1]
    >>> c.cid
    -1
    >>> x = {'call': 'foo', 'location': 'bar', 'states': [1]}
    >>> d = Call(js=x)
    >>> d
    Call(cid=-1, call='foo', location='bar', states=[1])
    >>> c == d
    True
    >>> c == Call(cid=10, call='foo', location='bar', states=[1])
    False
    """

    def __init__(self, call=None, location=None, states=None, js=None, cid=-1):
        if js is None:
            self.js = {'call': '', 'location': '', 'states': []}
        else:
            self.js = js
        if call is not None:
            self.js['call'] = call
        if location is not None:
            self.js['location'] = location
        if states is not None:
            self.js['states'] = states
        self.cid = cid

    def __eq__(self, other):
        return self.cid == other.cid and self.call == other.call \
            and self.location == other.location and self.states == other.states

    def __hash__(self):
        return hash((self.call, self.location, tuple(self.states if self.states is not None else ())))

    def as_string(self, include_state=True, include_location=True, state_sep=",", field_sep=":"):
        key = self.call
        if include_state and self.states is not None and len(self.states) > 0:
            key += field_sep + state_sep.join(map(str, self.states))
        if include_location and self.location is not None:
            key += field_sep + self.location
        return key

    def __repr__(self):
        return 'Call(cid=%r, call=%r, location=%r, states=%r)' % (self.cid, self.call, self.location, self.states)

    @property
    def call(self):
        return self.js['call']

    @call.setter
    def call(self, value):
        self.js['call'] = value

    @property
    def location(self):
        return self.js['location']

    @location.setter
    def location(self, value):
        self.js['location'] = value

    @property
    def states(self):
        return self.js['states']

    @states.setter
    def states(self, save):
        self.js['states'] = save




if __name__ == "__main__":
    import doctest
    doctest.testmod()

