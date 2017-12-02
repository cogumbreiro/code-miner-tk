#!/usr/bin/env python3
import shutil
import bz2
import glob
import itertools
import os
import ijson

def find_files(dirname, ext):
    return glob.glob(os.path.join(dirname, "**", ext), recursive=True)

def find_sal(dirname):
    return itertools.chain(
        find_files(dirname, "*.sal"),
        find_files(dirname, "*.sal.bz2")
    )

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Merges multipe Salento JSON Package files into a Salento JSON Dataset.")
    parser.add_argument("-i", dest="infile", nargs='?', type=argparse.FileType('r'),
                     default=sys.stdin, help="A list of filenames each pointing to Salento JSON Package. Default: standard input.")
    
    parser.add_argument("-d", dest="dir", nargs='?', type=str,
                     default=None, help="A directory containing Salento JSON Package. Default: standard input.")

    parser.add_argument("-o", dest="outfile", nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout, help="A Salento JSON Dataset. Defaut: standard output.")

    parser.add_argument("-s", help="Skip malformed input.", dest="skip",
                     action="store_true")
    args = parser.parse_args()

    print("{\"packages\": [", file=args.outfile)
    add_comma = False

    all_files = args.infile if args.dir is None else find_sal(args.dir)
    import json
    for x in all_files:
        x = x.strip()

        fopen = bz2.open if x.endswith(".bz2") else open
        if args.skip:
            with fopen(x, "rt") as fd:
                try:
                    for seq in ijson.items(fd, 'foo'):
                        pass
                except:
                    print("Error parsing file " + x, file=sys.stderr)
                    continue
        
        if add_comma:
            print(",",  file=args.outfile)
        else:
            add_comma = True

        with fopen(x, "rt") as fd:
            shutil.copyfileobj(fd, args.outfile)
            
    print("]}", file=args.outfile)
    
    
if __name__ == '__main__':
    main()
