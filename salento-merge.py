#!/usr/bin/env python3
import shutil

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Merges multipe Salento JSON Package files into a Salento JSON Dataset.")
    parser.add_argument("-i", dest="infile", nargs='?', type=argparse.FileType('r'),
                     default=sys.stdin, help="A list of filenames each pointing to Salento JSON Package. Default: standard input.")
    parser.add_argument("-o", dest="outfile", nargs='?', type=argparse.FileType('w'),
                     default=sys.stdout, help="A Salento JSON Dataset. Defaut: standard output.")
    args = parser.parse_args()

    print("{\"packages\": [", file=args.outfile)
    add_comma = False
    for x in args.infile:
        if add_comma:
            print(", ")
        else:
            add_comma = True
        with open(x.strip()) as fp:
            shutil.copyfileobj(fp, sys.stdout)
            
    print("]}", file=args.outfile)
    
    
if __name__ == '__main__':
    main()
