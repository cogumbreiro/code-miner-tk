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
import shutil
import bz2
import glob
import itertools
import os
import ijson
import json

def write_packages(fd, args):
    if args.packages:
        for pkg in ijson.items(fd, 'packages.item'):
            yield
            json.dump(pkg, args.outfile)
    else:
        yield
        shutil.copyfileobj(fd, args.outfile)

def get_fds(files, skip):
    for x in files:
        x = x.strip()
        try:
            fopen = bz2.open if x.endswith(".bz2") else open

            if skip:
                with fopen(x, "rt") as fd:
                    try:
                        for seq in ijson.items(fd, 'foo'):
                            pass
                    except:
                        print("Error parsing file " + x, file=sys.stderr)
                        continue
            
            with fopen(x, "rt") as fd:
                yield fd
        except FileNotFoundError as err:
            print("File not found:", err, file=sys.stderr)

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Merges multipe Salento JSON Package files into a Salento JSON Dataset.")
    get_input_files = common.parser_add_input_files(parser)
    parser.add_argument("-o", dest="outfile", nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout, help="A Salento JSON Dataset. Defaut: standard output.")
    parser.add_argument("--packages", help="Each file has its own packages.", dest="packages",
                     action="store_true")
    parser.add_argument("-s", help="Skip malformed input.", dest="skip",
                     action="store_true")
    args = parser.parse_args()

    print("{\"packages\": [", file=args.outfile)
    add_comma = False

    for fd in get_fds(get_input_files(args), args.skip):
        for _ in write_packages(fd, args):
            if add_comma:
                print(",",  file=args.outfile)
            else:
                add_comma = True
            

    print("]}", file=args.outfile)
    
    
if __name__ == '__main__':
    main()
