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

import json

from apisan.parse.event import CallEvent
from apisan.parse import explorer

__all__ = ['exec_tree_to_sequences']

def to_call_path(path):
    for node in path:
        evt = node.event
        if isinstance(evt, CallEvent):
            name = evt.call_name
            if name is not None:
                yield {'call':name, 'states':[], 'location':evt.code}

def exec_tree_to_sequences(exec_tree):
    """
    Converts an `ExecTree` to a generator of Salento sequences
    """
    for path in exec_tree:
        call_path = tuple(to_call_path(path))
        if len(call_path) > 0:
            yield {'sequence':call_path}


def parse_file(filename):
    """
    Parses an APISAN file as a Salento object tree.
    """
    visited = set()
    for tree in explorer.parse_file(filename):
        for seq in exec_tree_to_sequences(tree):
            tid = tuple(c['call'] for c in seq['sequence'])
            if tid not in visited:
                visited.add(tid)
                yield seq

def convert_to_json(filename, out):
    out.write('{"data":[')
    first = True
    for seq in parse_file(filename):
        if first:
            first = False
        else:
            out.write(',')
        json.dump(seq, out)
    out.write('],"name":')
    json.dump(filename, out)
    out.write("}")


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Converts a APISAN file format into a Salento JSON Package file.")
    parser.add_argument("-i", dest="infile", nargs='?', type=str,
                     default="/dev/stdin", help="A filename of the APISAN file format (.as). Default: /dev/stdin.")
    parser.add_argument("-o", dest="outfile", nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout, help="A Salento JSON Package file format. Defaut: standard output.")
    args = parser.parse_args()

    convert_to_json(args.infile, args.outfile)

if __name__ == '__main__':
    main()

