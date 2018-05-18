#!/usr/bin/env python3
import os.path
import sys
import argparse
import json

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

import sal
import common

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='The input JSON file.')
    parser.add_argument('outfile', nargs='?', default=None, help='The output file. Default: standard-output')
    args = parser.parse_args()

    try:

        with common.smart_open(args.infile, 'rt') as f:
            data = json.load(f)

        ds = sal.Dataset(js=data)
        ds.flatten_sequences()

        if args.outfile is None:
            json.dump(data, sys.stdout)
        else:
            with common.smart_open(args.outfile, 'wt') as f:
                json.dump(data, f)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == '__main__':
    main()
