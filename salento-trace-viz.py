#!/usr/bin/env python3
import json
import sys
try:
    import graphviz
except ImportError as e:
    print("ERROR:", e.args[0], file=sys.stderr)
    print("\nInstal the needed dependencies with:\n\tpip3 install graphviz", file=sys.stderr)
    sys.exit(1)

import argparse

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

def sequence_at(seq, idx, name, arg='location'):
    return match(seq['sequence'][idx][arg], name)

def sequence_ends_with(seq, name, arg='location'):
    return match(seq['sequence'][-1][arg], name)

def sequence_matches(seq, name, arg='location'):
    name = name.lower()
    for call in seq['sequence']:
        if match(call[arg], name):
            return True
    return False

class GraphBuilder:
    def __init__(self, args):
        self.graph = graphviz.Digraph()
        self.ids = dict()
        self.known_edges = set()
        self.args = args

    def node(self, name):
        if name in self.ids:
            return self.ids[name]
        else:
            idx = str(len(self.ids))
            self.ids[name] = idx
            self.graph.node(idx, label=name)
            return idx

    def call(self, call):
        return self.node(call_name(call))

    def edge(self, left, right):
        edge = self.call(left), self.call(right)
        if edge not in self.known_edges:
            self.known_edges.add(edge)
            self.graph.edge(*edge)

    def on_start_seq(self):
        self.last_call = None

    def on_call(self, call):
        self.call(call)
        if self.last_call is not None:
            self.edge(self.last_call, call)
        self.last_call = call

    def on_end_seq(self):
        self.last_call = None

    def translate_sequence(self, seq):
        if self.args.end is not None and not sequence_at(seq, idx=-1, name=self.args.end):
            return
        if self.args.match is not None and not sequence_matches(seq, name=self.args.match):
            return
        if self.args.start is not None and not sequence_at(seq, idx=0, name=self.args.start):
            return
        self.on_start_seq()
        for call in get_calls(seq=seq):
            self.on_call(call)
        self.on_end_seq()

    def translate_package(self, pkg):
        idx = 0
        for seq in get_sequences(pkg=pkg):
            self.translate_sequence(seq)
            idx += 1

    def build(self, doc):
        for pkg in get_packages(doc):
            self.translate_package(pkg)

def salento_to_trace(args, doc):
    b = GraphBuilder(args)
    b.build(doc)
    return b.graph


def get_sequences(pkg):
    return pkg['data']

def get_calls(seq):
    return seq['sequence']

def get_packages(doc):
    if "packages" in doc:
        for pkg in doc['packages']:
            yield pkg
    else:
        yield doc

def show_nth(doc, nth):
    visited = set()
    for pkg in get_packages(doc):
        for seq in get_sequences(pkg):
            calls = get_calls(seq)
            if len(calls) > 0:
                call = call_name(calls[nth])
                if call not in visited:
                    print(call)
                    visited.add(call)

def main():
    parser = argparse.ArgumentParser(description="""Salento Graphviz trace vizualizer.
The input (a Salento JSON dataset) can be filtered by its terms (the call name \n
and location joined with a colon `:`); the switches are combined with
a logical AND, thu prunning more the search.

You can use `^` and `$` to represent the beginning and the end of a term,
respectively.
Example: `oo()` matches any call name OR any location that contains `oo()`.
Example: `^foo` matches any call name that starts with `foo`.
Example: `:30$` matches any location that ends with `:30`.
Example: `^foo():file.c:30` matches any term that starts with a call name
`foo()` and a location `file.c:30`.
""")
    parser.add_argument('filename', help='input data file')
    match_help = " We"
    parser.add_argument('--match', '-m', help='Filter in sequences that contain the given location.')
    parser.add_argument('--end', '-e', help='Filter in sequences that end with the given location.')
    parser.add_argument('--start', '-s', help='Filter in sequences that start with the given location.')
    parser.add_argument('--list-first', action='store_true', help="List the first tokens of the dataset.")
    parser.add_argument('--list-last', action='store_true', help="List the last tokens of the dataset.")
    parser.add_argument('--outfile', '-o', default=sys.stdout, help="Save the Graphviz file. Default: standard output.")
    args = parser.parse_args()
    js = json.load(open(args.filename))
    if args.list_first:
        show_nth(js, 0)
    elif args.list_last:
        show_nth(js, -1)
    else:
        g = salento_to_trace(args, js)
        if args.outfile is sys.stdout:
            print(g.source)
        else:
            g.save(filename=args.outfile)

if __name__ == '__main__':
    main()
