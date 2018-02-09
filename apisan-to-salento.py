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

from apisan.parse.event import CallEvent, AssumeEvent
from apisan.parse.symbol import *
from apisan.parse import explorer
from collections import OrderedDict

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
        return {'call':evt.call_name, 'states':states, 'location':evt.code}

def to_call_path(path):
    db = ArgsDB()
    for node in path:
        evt = node.event
        if isinstance(evt, CallEvent):
            yield from db.push_call(evt)
        elif isinstance(evt, AssumeEvent):
            yield from db.push_branch(evt)
        else:
            # flush
            yield from db.flush()
    yield from db.flush()

def to_call_path_simple(path):
    for node in path:
        evt = node.event
        if isinstance(evt, CallEvent):
            name = evt.call_name
            if name is not None:
                if last_node is not None:
                    yield evt_to_json(last_node)

def exec_tree_to_sequences(exec_tree):
    """
    Converts an `ExecTree` to a generator of Salento sequences
    """
    for path in exec_tree:
        call_path = tuple(to_call_path(path))
        if len(call_path) > 0:
            yield {'sequence':call_path}

def call_unique_id(call):
    return call['call'] + "".join(str(s) for s in call.get('states', []))

def sequence_unique_id(seq):
    return "".join(call_unique_id(c) for c in seq['sequence'])

def parse_file(filename):
    """
    Parses an APISAN file as a Salento object tree.
    """
    visited = set()
    for tree in explorer.parse_file(filename):
        for seq in exec_tree_to_sequences(tree):
            tid = sequence_unique_id(seq)
            if tid not in visited:
                visited.add(tid)
                yield seq

def convert_to_json(in_fname, out_fname, enclose_in_packages):
    with common.smart_open(out_fname, 'wt') as out:
        if enclose_in_packages:
            out.write('{"packages":[')
        out.write('{"data":[')
        first = True
        for seq in parse_file(in_fname):
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
                     default=sys.stdout, help="A Salento JSON Package file format. Defaut: standard output.")
    parser.add_argument("--packages", action="store_true", help="Outputs a Salento JSON packages format instead.")
    args = parser.parse_args()

    convert_to_json(args.infile, args.outfile, args.packages)

if __name__ == '__main__':
    main()

