#!/usr/bin/env python3
import ijson
import sys
import itertools
import glob
import os
import bz2

def find_files(dirname, ext):
    return glob.glob(os.path.join(dirname, "**", ext), recursive=True)

def find_sal(dirname):
    return itertools.chain(
        find_files(dirname, "*.sal"),
        find_files(dirname, "*.sal.bz2")
    )

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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Converts a Salento JSON dataset into plain text.")
    parser.add_argument("-i", dest="infile", nargs='?', type=str,
                     default="/dev/stdin", help="A file of the Salento Dataset format.")
    parser.add_argument("-d", dest="dir", nargs='?', type=str,
                     default=None, help="A directory containing Salento JSON Package. Default: standard input.")
    parser.add_argument("-o", dest="outfile", nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout, help="A file of a text format.")
    args = parser.parse_args()

    infiles = [args.infile]
    include_pkgs = True
    if args.dir is not None:
        infiles = find_sal(args.dir)
        include_pkgs = False

    for f in infiles:
        fopen = bz2.open if f.endswith(".bz2") else open
        with fopen(f, 'rb') as fp:
            try:
                for seq in sequences(fp, include_packages=include_pkgs):
                    if len(seq) > 0:
                        args.outfile.write(" ".join(seq) + " $END\n")
            except (ijson.common.IncompleteJSONError, OSError, IOError):
                print("Error parsing: " + f, file=sys.stderr)
                continue


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass


