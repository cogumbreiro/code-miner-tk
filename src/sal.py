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

class VDataset(IDataset):
    """
    >>> d = VDataset({'packages': [{'name': 'foo', 'data': []}]})
    >>> len(d)
    1
    >>> d[0] == Package([], name='foo', pid=0)
    True
    >>> 
    """
    def __init__(self, js):
        self.js = js

    def __iter__(self):
        for pid, pkg in enumerate(self.js['packages']):
            yield self.make_package(pkg, pid=pid)
    
    def __len__(self):
        return len(self.js['packages'])

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
    def from_js(cls, js, lazy=False):
        if lazy:
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

    def matches_at(self, name, idx, key):
        return match(key(self[idx]), name)

    def matches_any(self, name, key):
        name = name.lower()
        for call in self:
            if match(key(call), name):
                return True
        return False

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

    Slices work as expected:

        >>> x[:]
        [Call(cid=0, call='foo', location='bar', states=[1]), Call(cid=1, call='foo2', location='bar2', states=[])]

    Len also works as expected:

        >>> len(x)
        2

    Get-item also works:

        >>> x[0] == c
        True

    You can also retreive the last location of a sequence:

        >>> x.last_location
        'bar2'
    """
    def __init__(self, js, sid=-1):
        self.sid = sid
        self.js = js
    
    def __iter__(self):
        for cid, call in enumerate(get_calls(self.js)):
            yield VCall(cid=cid, js=call)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return from_slice(key, self)
        else:
            return VCall(cid=key, js=get_calls(self.js)[key])

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
    
    @property
    def location(self):
        return self.js['location']
    
    @property
    def states(self):
        return self.js['states']

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

def filter_unknown_vocabs(json_data, vocabs=None, stopwords=set(), min_seq_len=3, branch_tokens=set(['$BRANCH'])):
    """
    By default sequences with 2 or fewer are filtered out.

        >>> small_seq = Sequence([Call('foo'), Call('bar')])
        >>> pkg = Package([small_seq], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js)
        >>> len(pkg)
        0

    If we change the set the minimum size to 0, we do not filter based on lenght:
    
        >>> small_seq = Sequence([Call('foo'), Call('bar')])
        >>> pkg = Package([small_seq], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, min_seq_len=0)
        >>> len(pkg)
        1

    We can use stop words to eliminate calls, in this case by removing
    the call 'foo' we actually remove the first sequence (as it falls below
    the acceptable minimum length):

        >>> seq1 = Sequence([Call('foo'), Call('bar')])
        >>> seq2 = Sequence([Call('foo'), Call('bar'), Call('baz')])
        >>> pkg = Package([seq1, seq2], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, stopwords=['bar'], min_seq_len=2)
        >>> len(pkg)
        1
        >>> len(pkg[0])
        2
        >>> pkg[0][0].call == 'foo' and pkg[0][1].call == 'baz'
        True

    We can vocabs to limit the accepted terms, in this case by removing
    the call 'bar' (note that we are not filtering out based on minimum length):

        >>> seq1 = Sequence([Call('foo'), Call('bar')])
        >>> seq2 = Sequence([Call('foo'), Call('bar'), Call('baz')])
        >>> pkg = Package([seq1, seq2], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, vocabs=['foo', 'baz'], min_seq_len=0)
        >>> len(pkg)
        2
        >>> len(pkg[0])
        1
        >>> pkg[0][0].call
        'foo'
        >>> len(pkg[1])
        2
        >>> pkg[1][0].call, pkg[1][1].call
        ('foo', 'baz')
        
    By default there's a notion of a branch token; when a non-branch token is
    removed (because it is a stop word or because it is not in the vocabs),
    all succeeding branch tokens are removed. In the following example we have
    two branch tokens that are removed because 'foo' is removed.

        >>> seq1 = Sequence([Call('foo'), Call('X'), Call('Y'), Call('bar')])
        >>> pkg = Package([seq1], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, stopwords=['foo'], min_seq_len=0, branch_tokens=('X','Y'))
        >>> len(pkg)
        1
        >>> len(pkg[0])
        1
        >>> pkg[0][0].call == 'bar'
        True
    """
    def check_seq(seq):
        allow_term = vocabs.__contains__ if vocabs is not None else lambda x: True
        events = []
        to_remove = False
        for x in seq['sequence']:
            # This branch is needed because if we remove a term, we must remove
            # the consecutive $BRANCH tokens if they exist
            if to_remove:
                if x['call'] in branch_tokens:
                    continue
                to_remove = False
            call = x['call']
            to_remove = not allow_term(call) or call in stopwords
            if not to_remove:
                events.append(x)

        seq['sequence'] = events
        return len(events) >= min_seq_len

    for pkg in get_packages(doc=json_data):
        pkg['data'] = list(filter(check_seq, pkg['data']))



if __name__ == "__main__":
    import doctest
    doctest.testmod()

