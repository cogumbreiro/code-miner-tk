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

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Merges multipe Salento JSON Package files into a Salento JSON Dataset.")
    get_input_files = common.parser_add_input_files(parser)
    parser.add_argument("-o", dest="outfile", nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout, help="A Salento JSON Dataset. Defaut: standard output.")

    parser.add_argument("-s", help="Skip malformed input.", dest="skip",
                     action="store_true")
    args = parser.parse_args()

    print("{\"packages\": [", file=args.outfile)
    add_comma = False

    import json
    for x in get_input_files(args):
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
