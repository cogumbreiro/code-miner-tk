#!/usr/bin/env python3
import copy
import math
import tempfile
import json
import os.path
import itertools

try:
    import common
except ImportError:
    import sys
    import os
    from os import path
    home = path.abspath(path.dirname(sys.argv[0]))
    sys.path.append(path.join(home, "src"))
    import common

def sizeof(pkg):
    return len(pkg['data'])

def split(pkg, count):
    result = copy.deepcopy(pkg)
    del result['data'][count:]
    del pkg['data'][0:count]
    return result

def do_partition(packages, nprocs):
    """
    >>> list(do_partition({
    ...    'packages': [
    ...        {'data': [1,2,3,4,5], 'name': 'a'},
    ...        {'data': [6,7,8], 'name': 'b'}
    ...    ]
    ... }, 3))
    [(0, {'data': [1, 2, 3], 'name': 'a'}), (1, {'data': [4, 5], 'name': 'a'}), (1, {'data', [6], 'name': 'b'}), (2, {'data': [7, 8], 'name': 'b'})]
    """
    total = sum(sizeof(pkg) for pkg in packages['packages'])
    per_task = math.ceil(total / nprocs)
    idx = 0
    consumed = 0
    for pkg in packages['packages']:
        if consumed + sizeof(pkg) <= per_task:
            yield idx, pkg
            consumed += sizeof(pkg)
        else:
            while consumed + sizeof(pkg) > per_task:
                yield idx, split(pkg, per_task - consumed)
                consumed = 0
                idx += 1
            if sizeof(pkg) > 0:
                yield idx, pkg
                consumed += sizeof(pkg)

def partition(packages, nprocs):
    for (_, elems) in itertools.groupby(do_partition(packages, nprocs), lambda x: x[0]):
        yield (x[1] for x in elems)

def partition_files(packages, files, print_filename):
    for (fname, pkgs) in zip(files, partition(packages, len(files))):
        with common.smart_open(fname, "wt") as fp:
            fp.write('{"packages": [')
            is_first = True
            for pkg in pkgs:
                if not is_first:
                    fp.write(',')
                json.dump(pkg, fp)
                is_first = False
            fp.write(']}')
            if print_filename:
                print(fname)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Partition a Salento input file.")
    parser.add_argument("filename", type=str,
                    help="The JSON filename we are processing.")
    parser.add_argument("--format", type=str, default="{basename}-{idx}.json{compress}", help="Output filename template. Default: %(default)s")
    parser.add_argument("-j", action="store_true", help="Compress data.")
    parser.add_argument("-v", action="store_true", help="Print filename.")
    get_nprocs = common.parser_add_parallelism(parser)
    args = parser.parse_args()

    nprocs = args.nprocs
    basename, ext = os.path.splitext(args.filename)
    while ext != "":
        basename, ext = os.path.splitext(basename)
    filenames = [args.format.format(basename=basename, idx=idx, compress=".bz2" if args.j else "") for idx in range(nprocs)]
    with common.smart_open(args.filename, 'rt') as fp:
        partition_files(json.load(fp), filenames, print_filename=args.v)

if __name__ == "__main__":
    main()
