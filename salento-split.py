#!/usr/bin/env python3
import copy
import math
import tempfile
import json
import os.path
import itertools
from operator import *
import sys

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
else:
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

import common
import sal

def sizeof(pkg):
    return len(pkg['data'])

def split(pkg, count):
    result = copy.deepcopy(pkg)
    del result['data'][count:]
    del pkg['data'][0:count]
    return result

def foreach_sequence(ds):
    for pkg in sal.get_packages(ds):
        pkg_name = sal.get_package_name(pkg=pkg)
        for seq in sal.get_sequences(pkg=pkg):
            yield pkg_name, seq

def assemble(seqs):
    elems = itertools.groupby(seqs, itemgetter(0))
    return list(map(lambda row: {'name':row[0], 'data': list(map(itemgetter(1), row[1]))}, elems))


def partition(packages, count):
    """
    >>> list(map(list, partition({
    ...    'packages': [
    ...        {'data': [1,2,3,4,5], 'name': 'a'},
    ...        {'data': [6,7,8], 'name': 'b'}
    ...    ]
    ... }, 3)))
    [[{'name': 'a', 'data': [1, 2, 3]}], [{'name': 'a', 'data': [4, 5]}, {'name': 'b', 'data': [6]}], [{'name': 'b', 'data': [7, 8]}]]

    """
    total = sum(sizeof(pkg) for pkg in packages['packages'])
    per_task = math.ceil(total / count)
    counts = itertools.repeat(per_task, count)
    elems = common.partition_iter(foreach_sequence(packages), counts)
    return map(assemble, elems)

def divide(packages, ratio=.8):
    """
    >>> list(map(list, divide({
    ...    'packages': [
    ...        {'data': [1,2,3,4,5], 'name': 'a'},
    ...        {'data': [6,7,8,9,10], 'name': 'b'}
    ...    ]
    ... })))
    [[{'name': 'a', 'data': [1, 2, 3, 4, 5]}, {'name': 'b', 'data': [6, 7, 8]}], [{'name': 'b', 'data': [9, 10]}]]

    """
    if ratio > 1.0 or ratio < 0.0:
        raise ValueError("Expecting betweeen 0..1, but got %r" % ratio)
    total = sum(sizeof(pkg) for pkg in packages['packages'])
    ratio1 = ratio
    ratio2 = 1 - ratio
    counts = (math.ceil(ratio1 * total), math.ceil(ratio2 * total))
    elems = common.partition_iter(foreach_sequence(packages), counts)
    return map(assemble, elems)

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

def partition_by_count(js, filenames, args):
    for filename, pkgs in zip(filenames, partition(js, len(filenames))):
        write_packages(filename, pkgs)
        yield filename


def partition_by_package(js, filenames, args):
    for filename, pkg in zip(filenames, js['packages']):
        write_packages(filename, [pkg])
        yield filename

def partition_by_ratio(js, filenames, args):
    for filename, pkgs in zip(filenames, divide(js, args.ratio)):
        write_packages(filename, pkgs)
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
    group.add_argument('--ratio', type=float, help='Partition the dataset into 2 parts, according to the ratio given by this argument.')
    args = parser.parse_args()

    basename, ext = os.path.splitext(args.filename)
    while ext != "":
        basename, ext = os.path.splitext(basename)

    with common.smart_open(args.filename, 'rt') as fp:
        js = json.load(fp)
        if args.n_ways is not None:
            count = args.n_ways
        elif args.ratio is not None:
            count = 2
        else:
            count = len(js['packages'])
        
        filenames = [args.format.format(basename=basename, idx=idx, compress=".bz2" if args.j else "") for idx in range(count)]

        if args.n_ways is not None:
            part_algo = partition_by_count
        elif args.ratio is not None:
            part_algo = partition_by_ratio
        else:
            part_algo = partition_by_package

        for fname in part_algo(js, filenames, args):
            if args.v:
                print(fname)


if __name__ == "__main__":
    main()
