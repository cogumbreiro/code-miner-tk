#!/usr/bin/env python3

# PREAMBLE TO LOAD APISAN

try:
    import apisan
except ImportError:
    import sys
    import os
    from os import path
    apisan_home = path.join(path.dirname(sys.argv[0]), 'apisan')
    apisan_home = os.environ.get('APISAN_HOME', apisan_home)
    sys.path.append(path.join(apisan_home, 'analyzer'))

try:
    import apisan
except ImportError:
    import sys
    sys.stderr.write("apisan not found! Download apisan and set variable `APISAN_HOME` to the repository location.\n")
    sys.exit(-1)

# PREAMBLE TO LOAD COMMON

try:
    import common
except ImportError:
    import sys
    import os
    from os import path
    home = path.abspath(path.dirname(sys.argv[0]))
    sys.path.append(path.join(home, "src"))

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

def make_call(name, location, states=()):
    assert name is not None
    assert location is not None
    return {'call':name, 'states':states, 'location': location}

def event_translator(evt, states=()):
    assert evt.call_name is not None
    assert evt.code is not None
    return make_call(name=evt.call_name, states=states, location=evt.code)

def navigate_paths(path, handler=lambda x, y: None):
    last_node = None
    stack = []
    row = []
    for node in path:
        if is_call(node):
            stack.append((node, row))
            row = []
            last_node = None
            
        elif is_return(node):
            new_node, new_row = stack.pop()
            if new_node.code == node.code and new_node.call_name == node.call_name:
                if len(row) > 0:
                    yield row
            
            row = new_row
            node = new_node
            last_node = node
            row.append(make_call(name=node.call_name, location=node.code))
        else:
            result = handler(last_node, node)
            if result is not None:
                row.append(result)

    if len(row) > 0:
        yield row

def to_call_path_branch(path):
    def on_assume(last_node, node):
        if isinstance(node, AssumeEvent) and last_node is not None and last_node.code is not None:
            return make_call(name="$BRANCH", location=last_node.code)
        
    return navigate_paths(path, on_assume)


def to_call_path_branch1(path):
    last_node = None
    for node in path:
        if isinstance(node, AssumeEvent) and last_node is not None and last_node.code is not None:
            yield make_call(name="$BRANCH", location=last_node.code)
        if is_call(node):
            yield event_translator(node)
            last_node = node
        else:
            last_node = None
            

def to_call_path_states(path):
    db = ArgsDB()
    for evt in path:
        if is_call(evt):
            yield from db.push_call(evt)
        elif isinstance(evt, AssumeEvent):
            yield from db.push_branch(evt)
        else:
            # flush
            yield from db.flush()
    yield from db.flush()


def to_call_path_simple(path):
    last_node = None
    stack = []
    row = []
    for node in path:
        if is_call(node):
            stack.append((node, row))
            row = []
            last_node = None
            
        elif is_return(node):
            new_node, new_row = stack.pop()
            if new_node.code == node.code and new_node.call_name == node.call_name:
                if len(row) > 0:
                    yield row
            
            row = new_row
            node = new_node
            last_node = node
            row.append(make_call(name=node.call_name, location=node.code))

    if len(row) > 0:
        yield row

class Translator(Enum):
    BASIC = partial(to_call_path_simple)
    BRANCH = partial(to_call_path_branch)
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
            #yield all_but_last(map(operator.attrgetter("event"), path))

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

