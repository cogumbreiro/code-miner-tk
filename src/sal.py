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

def eq_iter(l, r):
    if len(l) != len(r):
        return False

    for x,y in zip(l, r):
        if x != y: return False
    return True

class IDataset:
    __eq__ = eq_iter

    @property
    def count(self):
        return len(self)

    def __repr__(self):
        return repr(list(iter(self)))

    def foreach_call(self):
        for seq in self.foreach_sequence():
            for call in seq:
                yield call

    def foreach_sequence(self):
        for pkg in self:
            for seq in pkg:
                yield seq

    def translate_calls(self, alias):
        """
        We can supply a map of aliases; the terms are replaced before filtering.

            >>> seq1 = Sequence([Call('baz'), Call('X'), Call('Y'), Call('bar')])
            >>> pkg = Package([seq1], name='p')
            >>> ds = Dataset([pkg])
            >>> ds = VDataset(ds.js)
            >>> ds.translate_calls({'baz': 'foo', 'bar': 'ZZZ'})
            >>> len(ds)
            1
            >>> len(ds[0])
            1
            >>> list(ds[0][0].terms)
            ['foo', 'X', 'Y', 'ZZZ']
        """
        for term in self.foreach_call():
            call = term.call
            if call in alias:
                term.call = alias[call]

    def filter_calls(self, vocabs=None, stopwords=set(), branch_tokens=set(['$BRANCH']), call_filter=None):
        """
        There's a notion of a branch token: when a non-branch token is
        removed (because it is a stop word or because it is not in the vocabs),
        all succeeding branch tokens are removed. In the following example we have
        two branch tokens that are removed because 'foo' is removed.

            >>> seq1 = Sequence([Call('foo'), Call('X'), Call('Y'), Call('bar')])
            >>> pkg = Package([seq1], name='p')
            >>> ds = Dataset([pkg])
            >>> ds = VDataset(ds.js)
            >>> ds.filter_calls(stopwords=['foo'], branch_tokens=('X','Y'))
            >>> len(ds), len(ds[0])
            (1, 1)
            >>> list(ds[0][0].terms)
            ['bar']
        """
        f = make_filter_call(stopwords=stopwords, vocabs=vocabs)
        if call_filter is not None:
            f = call_filter(f)
        do_filter = make_filter_branch(f, branch_tokens=branch_tokens)

        for pkg in get_packages(doc=self.js):
            for seq in get_sequences(pkg=pkg):
                seq['sequence'] = list(do_filter(seq))

    def filter_sequences(self, min_length=0):
        for pkg in self:
            pkg[:] = filter(lambda s: len(s) >= min_length, pkg)

class VDataset(IDataset):
    """
    >>> d = VDataset({'packages': [{'name': 'foo', 'data': []}]})
    >>> len(d)
    1
    >>> d[0] == Package([], name='foo', pid=0)
    True

    """
    def __init__(self, js):
        self.js = js

    def __iter__(self):
        for pid, pkg in enumerate(self.js['packages']):
            yield self.make_package(pkg, pid=pid)

    def __len__(self):
        return len(self.js['packages'])

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.js["packages"].__setitem__(key, map(attrgetter("js"), value))
        else:
            self.js[idx] = value.js

    def __getitem__(self, key):
        if isinstance(key, slice):
            return from_slice(key, self)
        else:
            return self.make_package(self.js['packages'][key], pid=key)

    def make_package(self, js, pid):
        return VPackage(js, pid)


class Dataset(IDataset, collections.UserList):
    """
    >>> Dataset([])
    []
    >>> js = {'packages': [{'name': 'foo', 'data': []}]}
    >>> Dataset.from_js(js).js == js
    True
    """
    def __init__(self, pkgs):
        self.data = pkgs

    @property
    def js(self):
        return {'packages': list(x.js for x in self)}

    @classmethod
    def from_js(cls, js, lazy=True, adapt_from_package=True):
        if lazy:
            # adapt single-package to a dataset
            if adapt_from_package and ("packages" not in js and "data" in js):
                js = {"packages": [js]}
            return VDataset(js)
        return cls(list(Package.from_js(pkg, pid) for pid, pkg in enumerate(js['packages'])))


class IPackage:

    def group_by_last_location(self):
        by_last = attrgetter("last_location")
        # remove empty sequences
        elems = sorted(filter(lambda seq: len(seq) > 0, self), key=by_last)
        # Group by last location
        return itertools.groupby(elems, by_last)

    @property
    def count(self):
        return len(self)

    def __eq__(self, other):
        return self.name == other.name and eq_iter(self, other)

    def __repr__(self):
        return 'Package(%r, name=%r, pid=%r)' % (list(iter(self)), self.name, self.pid)


class VPackage(IPackage):
    '''
    >>> js_c = {'call': 'foo', 'location': 'bar', 'states': [1]}
    >>> js_s = {'sequence': [js_c]}
    >>> s = Sequence.from_js(js_s)
    >>> js = {'name': 'baz', 'data': [js_s]}
    >>> pkg = VPackage(js)
    >>> pkg
    Package([Sequence([Call(cid=0, call='foo', location='bar', states=[1])], sid=0)], name='baz', pid=-1)
    >>> s = VSequence(js_s, sid=0)
    >>> pkg.name
    'baz'
    >>> len(pkg)
    1
    >>> pkg[0] == s
    True
    '''
    def __init__(self, js, pid=-1):
        self.js = js
        self.pid = pid

    @property
    def name(self):
        return self.js['name']

    def __iter__(self):
        for sid, seq in enumerate(get_sequences(pkg=self.js)):
            yield self.make_sequence(sid=sid, js=seq)

    def __len__(self):
        return len(self.js['data'])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return from_slice(key, self)
        else:
            return self.make_sequence(sid=key, js=get_sequences(self.js)[key])

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.js['data'].__setitem__(key, map(attrgetter("js"), value))
        else:
            self.js[idx] = value.js

    def make_sequence(self, js, sid):
        return VSequence(js, sid)

class Package(IPackage, collections.UserList):
    """
    A package has a name a list of call-sequences.
    >>> js_c = {'call': 'foo', 'location': 'bar', 'states': [1]}
    >>> js_s = {'sequence': [js_c]}
    >>> s = Sequence.from_js(js_s)
    >>> js = {'name': 'baz', 'data': [js_s]}
    >>> pkg = Package([], 'foo')
    >>> pkg
    Package([], name='foo', pid=-1)
    >>> pkg.name
    'foo'
    >>> list(pkg)
    []
    >>> pkg.pid
    -1
    >>> len(pkg)
    0
    >>> Package.from_js(js).js == js
    True
    """
    def __init__(self, sequences, name, pid=-1):
        self.pid = pid
        self.name = name
        self.data = sequences

    @property
    def js(self):
        return {
            'name': self.name,
            'data': list(x.js for x in self),
        }

    @classmethod
    def from_js(cls, js, pid=-1):
        name = js['name']
        seqs = list(Sequence.from_js(seq, sid=sid) \
            for sid, seq in enumerate(get_sequences(js)))
        return cls(name=name, sequences=seqs, pid=pid)


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

class make_filter_call:
    """
    We can use stop words to eliminate calls, in this case by removing
    the call 'foo' we actually remove the first sequence (as it falls below
    the acceptable minimum length):

        >>> f = make_filter_call(stopwords=['bar'])
        >>> f(Call('foo').js)
        True
        >>> f(Call('bar').js)
        False


    We can vocabs to limit the accepted terms, in this case by removing
    the call 'bar' (note that we are not filtering out based on minimum length):

        >>> f = make_filter_call(vocabs=['foo', 'baz'])
        >>> f(Call('foo').js)
        True
        >>> f(Call('box').js)
        False
        >>> f(Call('baz').js)
        True

    """
    def __init__(self, vocabs=None, stopwords=set()):
        self.stopwords = stopwords
        self.allow_term = vocabs.__contains__ if vocabs is not None else lambda x: True

    def __call__(self, term):
        call = term['call']
        return self.allow_term(call) and call not in self.stopwords


class ISequence:
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



class VSequence(ISequence):
    """
    Given some sequence object which we build from a JSON object:

        >>> js_c1 = {'call': 'foo', 'location': 'bar', 'states': [1]}
        >>> js_c2 = {'call': 'foo2', 'location': 'bar2', 'states': []}
        >>> js = {'sequence': [js_c1, js_c2]}
        >>> c = Call(cid=0, call='foo', location='bar', states=[1])
        >>> x = VSequence(js)
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

        >>> x[0:1] = [Call(call='one'), Call(call='two')]
        >>> list(y.call for y in x)
        ['one', 'two', 'foo2']


    """
    def __init__(self, js, sid=-1):
        self.sid = sid
        self.js = js

    def __iter__(self):
        for cid, call in enumerate(get_calls(self.js)):
            yield VCall(cid=cid, js=call)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self.js['sequence'].__setitem__(key, map(attrgetter("js"), value))
        else:
            self.js[idx] = value.js

    def __getitem__(self, key):
        if isinstance(key, slice):
            return from_slice(key, self)
        else:
            return VCall(cid=key, js=get_calls(self.js)[key])

    def __delitem__(self, key):
        del self.js[key]

    def __len__(self):
        return len(get_calls(self.js))


class Sequence(ISequence, collections.UserList):
    """
    A list of calls.

    Given a sequence of calls:

        >>> c = Call(call='foo', location='bar', states=[1])
        >>> x = Sequence([c])
        >>> x
        Sequence([Call(cid=-1, call='foo', location='bar', states=[1])], sid=-1)

    You can retreive the last location of a sequence:

        >>> x.last_location
        'bar'

    The length works as expected:

        >>> len(x)
        1

    Get-item also works:
        >>> x[0] == c
        True

    You can also convert into a JS object with the `js` field, and you
    can convert a JS object into a Sequence object with `from_js`:

        >>> js_c = {'call': 'foo', 'location': 'bar', 'states': [1]}
        >>> js = {'sequence': [js_c]}
        >>> VSequence(js) == Sequence.from_js(js)
        True
        >>> Sequence.from_js(js).js == js
        True
    """
    def __init__(self, calls, sid=-1):
        self.data = calls
        self.sid = sid

    @classmethod
    def from_js(cls, js, sid=-1):
        calls = list(Call.from_js(c, cid=cid) for cid, c in enumerate(get_calls(seq=js)))
        return cls(calls, sid=sid)

    @property
    def js(self):
        return {'sequence': list(x.js for x in self)}

class ICall:
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

class VCall(ICall):
    """
    >>> x = VCall({'call': 'foo', 'location': 'bar', 'states': [1]})
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
    """
    def __init__(self, js, cid=-1):
        self.cid = cid
        self.js = js

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


class Call(ICall):
    '''
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
    >>> d = Call.from_js(x)
    >>> d
    Call(cid=-1, call='foo', location='bar', states=[1])
    >>> c == d
    True
    >>> c == Call(cid=10, call='foo', location='bar', states=[1])
    False
    >>> Call.from_js(x).js == x
    True
    '''
    def __init__(self, call, location='', states=[], cid=-1):
        self.call = call
        self.location = location
        self.states = states
        self.cid = cid

    @classmethod
    def from_js(cls, js, cid=-1):
        return cls(js['call'], js['location'], js['states'], cid)

    @property
    def js(self):
        return {'call': self.call, 'location': self.location, 'states': self.states}




if __name__ == "__main__":
    import doctest
    doctest.testmod()

