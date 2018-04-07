#!/usr/bin/env python3
try:
    import common
except ImportError:
    import sys
    import os
    from os import path
    home = path.abspath(path.dirname(sys.argv[0]))
    sys.path.append(path.join(home, "src"))

import common

import ijson
import sys
import itertools
import glob
import os
import bz2
import subprocess
import shlex

class sequences:
    def __init__(self, fd, include_packages=True):
        self.fd = fd
        self.include_packages = include_packages

    def __iter__(self):
        query = 'data.item.sequence'
        if self.include_packages:
            query = 'packages.item.' + query
        for seq in ijson.items(self.fd, query):
            yield(list(x['call'] for x in seq))

def run_slow(f, eol=None, include_packages=None, **kwargs):
    fopen = bz2.open if f.endswith(".bz2") else open
    with fopen(f, 'rb') as fp:
        try:
            for seq in sequences(fp, include_packages=include_packages):
                if len(seq) > 0:
                    sys.stdout.write(" ".join(seq) + " " + eol + "\n")
        except (ijson.common.IncompleteJSONError, OSError, IOError):
            print("Error parsing: " + f, file=sys.stderr)

def run_acc(f, accelerator=None, eol=None, **kwargs):
    cmd = shlex.quote(accelerator)
    cmd += " --eol " + shlex.quote(eol)
    if f.endswith(".bz2"):
        cmd = "bzcat | " + cmd
    if subprocess.call("cat " + f + " | " + cmd, shell=True) != 0:
        print("Error parsing: " + f, file=sys.stderr)


def main():

    sal2txt = os.path.join(common.get_home(), 'sal2txt')
    import argparse
    parser = argparse.ArgumentParser(description="Converts a Salento JSON dataset into plain text.")
    get_input_files = common.parser_add_input_files(parser)
    
    parser.add_argument("-s", help="Set input format to Salento JSON Dataset format, otherwise expect Salento JSON Package format.", dest="include_pkgs",
                     action="store_true")
    parser.add_argument("--eol", help="Set the to be printed at the end of each sequence.", dest="eol", nargs="?", type=str,
                     default="$END")
    parser.add_argument("-a", dest="accelerator", nargs='?', type=str,
                     default=sal2txt, help="An accelerated program that converts Salento to text of a single file. Default: %(default)s.")
    args = parser.parse_args()

    if not os.path.isfile(args.accelerator):
        print("Warning: could not find accelerator program %r, falling back to pure Python, which is slower." % args.accelerator, file=sys.stderr)
    run = run_acc if os.path.isfile(args.accelerator) else run_slow

    for f in get_input_files(args):
        kwargs = dict(include_packages=args.include_pkgs, accelerator=args.accelerator, eol=args.eol)
        run(f, **kwargs)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass


