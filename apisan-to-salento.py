#!/usr/bin/env python3
import os
import sys

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
else:
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

# PREAMBLE TO LOAD APISAN
try:
    import apisan
except ImportError:
    apisan_home = os.path.join(CODE_MINER_HOME, 'apisan')
    apisan_home = os.environ.get('APISAN_HOME', apisan_home)
    sys.path.insert(0, os.path.join(apisan_home, 'analyzer'))

try:
    import apisan
except ImportError:
    sys.stderr.write("apisan not found! Download apisan and set variable `APISAN_HOME` to the repository location.\n")
    sys.exit(-1)

# PREAMBLE TO LOAD COMMON


import common

import json

from apisan.parse.event import *
from apisan.parse.symbol import *
from apisan.parse import explorer
from collections import OrderedDict
from enum import Enum, auto, unique
from functools import partial
import operator

def get_symbol(node):
    if isinstance(node, CallSymbol):
        for x in node.args:
            x = get_symbol(x)
            if x is not None:
                return x
    elif isinstance(node, BinaryOperatorSymbol):
        result = get_symbol(node.lhs)
        if result is None:
            result = get_symbol(node.rhs)
        return result
    elif isinstance(node, ArraySymbol) or isinstance(node, FieldSymbol):
        return get_symbol(node.base)
    elif isinstance(node, ConstraintSymbol):
        return get_symbol(node.symbol)
    elif isinstance(node, IDSymbol):
        return node.id

def get_symbols(evt):
    if evt.call is not None and evt.call.kind == SymbolKind.Call:
        for idx, node in enumerate(evt.call.children):
            yield (get_symbol(node), idx)


class ArgsDB:
    """
    Encodes variable names and branches as a state.
    """
    def __init__(self):
        self.last_node = None
        self.db = []
    
    def push_call(self, evt):
        if evt.call_name is not None:
            yield from self.flush()
            self.last_node = evt

    def push_branch(self, evt):
        if self.last_node is not None and evt._cond.text.startswith(self.last_node.call_name):
            yield from self.flush(branch=True)

    def flush(self, *args, **kwargs):
        if self.last_node is not None:
            yield self.evt_to_json(self.last_node, *args, **kwargs)
        self.last_node = None

    def lookup(self, name):
        if name is not None:
            for scope_idx, scope in enumerate(reversed(self.db)):
                if name in scope:
                    return scope_idx, scope[name]
        return -1, -1

    def get_args(self, evt):
        args = OrderedDict(get_symbols(evt))
        for name in args:
            left, right = self.lookup(name)
            yield left
            yield right
        self.db.append(args)

    def evt_to_json(self, evt, branch=False):
        states = [1 if branch else 0] + list(self.get_args(evt))
        return event_translator(evt, states)

def is_call(node):
    return isinstance(node, CallEvent) and node.code is not None and node.call_name is not None

def is_return(node):
    return isinstance(node, ReturnEvent) and node.code is not None and node.call_name is not None

def make_call(name, location, states=[]):
    assert name is not None
    assert location is not None
    return {'call':name, 'states':states, 'location': location}

def event_translator(evt, states=()):
    assert evt.call_name is not None
    assert evt.code is not None
    return make_call(name=evt.call_name, states=states, location=evt.code)

def pp_event(evt):
    if is_call(evt):
        return "" + evt.call_name + ":" + evt.code + ";\n{"
    if is_return(evt):
        return "}" #"R(" + evt.call_name + ":" + evt.code + ")"
    if isinstance(evt, AssumeEvent):
        return "A"

def pp_path(path):
    return ";\n".join(x for x in map(pp_event, path) if x is not None)

import xml.etree.ElementTree as ET

class PathNavigator:
    def __init__(self):
        self.stack = []
        self.row = []

    def push_scope(self):
        self.stack.append(self.row)
        self.row = []

    def pop_scope(self):
        row = self.row
        self.row = self.stack.pop()
        return row

    def add(self, elem):
        self.row.append(elem)

    def pop_all(self):
        yield self.row
        self.row = []
        yield from self.stack
        self.stack = []

    @classmethod
    def navigate(cls, path, is_push, is_pop, consume_dangling=True):
        """
        No pushes or pops, we just return a single list which contains the input:

            >>> list(PathNavigator.navigate([1, 2, 3], is_push=lambda x:False, is_pop=lambda x: False))
            [[1, 2, 3]]
        
        An example of one push and one pop, yielding two paths and including the push token

            >>> list(PathNavigator.navigate(["1.1",True,"2.1","2.2",False,"1.2"], is_push=lambda x: x==True, is_pop=lambda x:x==False))
            [['2.1', '2.2'], ['1.1', True, '1.2']]

        An example of a single push and no matching pop, yielding no errors and no values        

            >>> list(PathNavigator.navigate(["1.1",True,"2.1","2.2"], is_push=lambda x: x==True, is_pop=lambda x:x==False, consume_dangling=False))
            []
        """
        stack = cls()
        for node in path:
            if is_push(node):
                stack.add(node)
                stack.push_scope()
            elif is_pop(node):
                scope = stack.pop_scope()
                if len(scope) > 0:
                    yield scope
            else:
                stack.add(node)
        if consume_dangling:
            # Consume all the pending tokens
            for row in stack.pop_all():
                if len(row) > 0:
                    yield row

def foreach_apisan_trail(path):
    return PathNavigator.navigate(path,
            is_push=lambda x: isinstance(x, CallEvent),
            is_pop=lambda x:isinstance(x, ReturnEvent),
            consume_dangling=True)

def process_assumes(path):
    """
    Given a path handles assume events; this function yields all events
    in the path, for each event it also yields the updates to the symbol table
    or None when there are no updates.
    """
    assumes = []
    last_node  = None
    for node in path:
        if isinstance(node, AssumeEvent):
            if last_node is not None and last_node.code is not None:
                assumes.append(node)
        elif is_call(node):
            if last_node is not None:
                yield last_node, assumes
            last_node = node
            assumes = []
        else:
            last_node = None

    if last_node is not None:
        yield last_node, assumes

def translate_path_branch(path):
    for node, assumes in process_assumes(path):
        yield make_call(name=node.call_name, location=node.code)
        if len(assumes) > 0:
            yield make_call(name="$BRANCH", location=node.code)

def translate_path_branch_states(path):
    for node, assumes in process_assumes(path):
        states = [1 if len(assumes) > 0 else 0]
        yield make_call(name=node.call_name, location=node.code, states=states)

def translate_path_simple(path):
    for node in path:
        if is_call(node):
            yield make_call(name=node.call_name, location=node.code)

def to_call_path_branch(path):
    for trail in foreach_apisan_trail(path):
        yield list(translate_path_branch(trail))

def to_call_path_branch_states(path):
    for trail in foreach_apisan_trail(path):
        yield list(translate_path_branch_states(trail))

def to_call_path_simple(path):
    for trail in foreach_apisan_trail(path):
        yield list(translate_path_simple(trail))

def translate_path_states(path):
    db = ArgsDB()
    for node, assumes in process_assumes(path):
        if is_call(evt):
            yield from db.push_call(evt)
        elif isinstance(evt, AssumeEvent):
            yield from db.push_branch(evt)
        else:
            # flush
            yield from db.flush()
    yield from db.flush()



class Translator(Enum):
    BASIC = partial(to_call_path_simple)
    BRANCH = partial(to_call_path_branch)
    BRANCH_STATES = partial(to_call_path_branch_states)
#    STATES = partial(lambda x: [list(to_call_path_states(x))])
    @classmethod
    def from_string(cls,s):
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError()
    def __str__(self):
        return self.name

    def __call__(self, *args):
        return self.value(*args)
   
def call_unique_id(evt):
    return evt['call'] + "".join(str(s) for s in evt.get('states', []))

def sequence_unique_id(seq):
    return "".join(call_unique_id(c) for c in seq['sequence'])

def all_but_last(elems):
    prev = None
    has_init = False
    for x in elems:
        if has_init:
            yield prev
        else:
            has_init = True
        prev = x

import xml.etree.ElementTree as ET
import sys

class Link:
    def __init__(self, elem, next=None):
        self.elem = elem
        self.next = next

    def __iter__(self):
        link = self
        while link is not None:
            yield link.elem
            link = link.next

def is_eop(node):
    return isinstance(node.event, EOPEvent)

def depth_first_search(tree):
    """
    Traverse the tree via DFS navigation.
    """
    stack = [Link(tree.root)]
    env = {}
    while len(stack) > 0:
        node = stack.pop()
        if is_eop(node.elem):
            row = list(iter(node))
            del row[0]
            row.reverse()
            yield row
        stack.extend(Link(x, node) for x in iter(node.elem))

def parse_events(filename):
    for tree in explorer.parse_file(filename):
        for path in depth_first_search(tree):
            yield map(operator.attrgetter("event"), path)

def translate_file(filename, trans):
    """
    Parses an APISAN file as a Salento object tree.
    """
    visited = set()
    for path in parse_events(filename):
        path = list(path)
        for call_path in trans(path):
            if len(call_path) == 0:
                continue
            seq = {'sequence':call_path}
            tid = sequence_unique_id(seq)
            if tid not in visited:
                visited.add(tid)
                yield seq

def convert_to_json(in_fname, out_fname, enclose_in_packages, trans):
    with common.smart_open(out_fname, 'wt') as out:
        if enclose_in_packages:
            out.write('{"packages":[')
        out.write('{"data":[')
        first = True
        for seq in translate_file(in_fname, trans):
            if first:
                first = False
            else:
                out.write(',')
            json.dump(seq, out)
        out.write('],"name":')
        json.dump(in_fname, out)
        out.write("}")
        if enclose_in_packages:
            out.write(']}')


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Converts a APISAN file format into a Salento JSON Package file.")
    parser.add_argument("-i", dest="infile", nargs='?', type=str,
                     default="/dev/stdin", help="A filename of the APISAN file format (.as). Default: /dev/stdin.")
    parser.add_argument("-o", dest="outfile", nargs='?', type=str,
                     default="/dev/stdout", help="A Salento JSON Package file format. Defaut: standard output.")
    parser.add_argument("--packages", action="store_true", help="Outputs a Salento JSON packages format instead.")
    parser.add_argument("--translator", "-t", type=Translator.from_string, choices=list(Translator), default=Translator.BRANCH)
    args = parser.parse_args()
    convert_to_json(args.infile, args.outfile, args.packages, args.translator)

if __name__ == '__main__':
    main()

