#!/usr/bin/env python3

try:
    import salento
except ImportError:
    import sys
    import os
    from os import path
    sys.path.append(path.abspath(path.dirname(sys.argv[0])))
    import salento

import sys
import collections
import json
import multiprocessing
import concurrent.futures

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reads a cluster file from the standard input and prints the top-k most frequent terms.")
    get_wc = salento.parser_add_wc_binary(parser)
    get_nprocs = salento.parser_add_parallelism(parser)
    parser.add_argument("--top-k", "-n", dest="top_k", nargs='?', type=int, default=5,
                     help="How many terms to show. Default: %(default)s")

    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=get_nprocs(args)) as executor:
        for line in sys.stdin:
            topk = collections.Counter()
            for wf in salento.run_word_freqs(executor, get_wc(args), json.loads(line)):
                for (k,v) in wf.items():
                    topk[k] += v
            json.dump(dict(topk.most_common(args.top_k)), sys.stdout)
            print()

if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        pass # No more output, close graciously
