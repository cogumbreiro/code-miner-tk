#!/usr/bin/env python3

import ijson
import sys

def prog2seq(fd):
    for seq in ijson.items(fd, 'packages.item.data.item.sequence'):
        yield(list(x['call'] for x in seq))

class sequences:
    def __init__(self, fd):
        self.fd = fd
    def __iter__(self):
        return prog2seq(self.fd)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Converts a Salento JSON dataset into plain text.")
    parser.add_argument("-i", dest="infile", nargs='?', type=argparse.FileType('rb'),
                     default=sys.stdin, help="A file of the Salento Dataset format.")
             
    parser.add_argument("-o", dest="outfile", nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout, help="A file of a text format.")
    args = parser.parse_args()

    for seq in sequences(args.infile):
        if len(seq) > 0:
            args.outfile.write(" ".join(seq) + " $END\n")

if __name__ == '__main__':
    main()


