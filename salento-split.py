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

def write_packages(filename, pkgs):
    with common.smart_open(filename, "wt") as fp:
        fp.write('{"packages": [')
        is_first = True
        for pkg in pkgs:
            if not is_first:
                fp.write(',')
            json.dump(pkg, fp)
            is_first = False
        fp.write(']}')

def partition_by_count(js, filenames):
    for filename, pkgs in zip(filenames, partition(js, len(filenames))):
        write_packages(filename, pkgs)
        yield filename


def partition_by_package(js, filenames):
    for filename, pkg in zip(filenames, js['packages']):
        write_packages(filename, [pkg])
        yield filename

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Partition a Salento input file.")
    parser.add_argument("filename", type=str,
                    help="The JSON filename we are processing.")
    parser.add_argument("--format", type=str, default="{basename}-{idx}.json{compress}", help="Output filename template. Default: %(default)s")
    parser.add_argument("-j", action="store_true", help="Compress data.")
    parser.add_argument("-v", action="store_true", help="Print filename.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--n-ways', type=int, help='Partition the dataset into a given number of files')
    group.add_argument('--per-package', action='store_true', help='Partition each package into a given file.')
    args = parser.parse_args()

    basename, ext = os.path.splitext(args.filename)
    while ext != "":
        basename, ext = os.path.splitext(basename)

    with common.smart_open(args.filename, 'rt') as fp:
        js = json.load(fp)
        if args.n_ways is not None:
            count = args.n_ways
        else:
            count = len(js['packages'])
        
        filenames = [args.format.format(basename=basename, idx=idx, compress=".bz2" if args.j else "") for idx in range(count)]

        part_algo = partition_by_count if args.n_ways is not None else partition_by_package

        for fname in part_algo(js, filenames):
            if args.v:
                print(fname)


if __name__ == "__main__":
    main()
