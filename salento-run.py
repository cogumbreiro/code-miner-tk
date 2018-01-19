#!/usr/bin/env python3
import copy
import math
import tempfile
import json
import os.path

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
    last_idx = -1
    is_first = True
    files = iter(list(tempfile.NamedTemporaryFile(delete=False, mode="wt", suffix=".json") for n in range(nprocs)))
    out = next(files)
    for (idx, pkg) in do_partition(packages, nprocs):
        if idx != last_idx:
            if idx != 0:
                out.write(']}')
                out.close()
                yield out.name
                out = next(files)
            is_first = True
            print('{"packages": [', file=out)
            last_idx = idx
        
        if not is_first:
            print(",", file=out)
        json.dump(pkg, out)
        is_first = False
    if last_idx != -1:
        out.write(']}')
        out.close()
        yield out.name

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Given an archive of C files, generates Salento Package JSON files.")
    parser.add_argument("-d", dest="data_dir", type=str,
                    required=not os.path.exists("save"),
                    default="save", help="The default Tensorflow model directory. DEFAULT: '%(default)s'")
    parser.add_argument("filename", type=str,
                    help="The JSON filename we are processing.")
    parser.add_argument("--salento-home", dest="salento_home", default=os.environ.get('SALENTO_HOME', None),
        required=os.environ.get('SALENTO_HOME', None) is None,
        help="The directory where the salento repository is located (defaults to $SALENTO_HOME). Default %(default)s")
    parser.add_argument("--aggregator", default="sequence", choices=['sequence', 'kld'], help="The aggregator to run. Default %(default)s")
    get_nprocs = common.parser_add_parallelism(parser)
    args = parser.parse_args()

    script = os.path.join(args.salento_home, "src/main/python/salento/aggregators/%s_aggregator.py")
    nprocs = get_nprocs(args)
    with finish(concurrent.futures.ThreadPoolExecutor(max_workers=nprocs)) as executor:
        for fname in partition(json.load(open(args.filename)), get_nprocs(args)):
            @executor.submit
            def run(script=script, data_dir=args.data_dir, fname=fname):
                common.run("python3 %s --model_dir %s --data_file %s", script, data_dir, fname, silent=False)

if __name__ == "__main__":
    main()
