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
import sal

import numpy as np

import concurrent.futures
import sys
import argparse
import random
import os
import pickle
import json

from collections import Counter

def seq_term_frequency(seq):
    c = Counter()
    for call in sal.get_calls(seq=seq):
        c[call['call']] += 1
    return c

class State:
    def __init__(self):
        self.counter = Counter()
    
    def __call__(self, other):
        self.counter += other
    
    def get(self):
        return self.counter

def get_term_frequency(doc, nprocs, seq_len_treshold=1):
    result = State()
    with common.finish(concurrent.futures.ThreadPoolExecutor(max_workers=nprocs), accumulator=result) as executor:
        for pkg in sal.get_packages(doc=doc):
            for seq in sal.get_sequences(pkg=pkg):
                if len(sal.get_calls(seq=seq)) >= seq_len_treshold:
                    executor.submit(seq_term_frequency, seq)
    return result.get()

def get_common_vocabs(tf, idf_treshold=0.0025):
    (_,largest), = tf.most_common(1)
    result = set(term for term, freq in tf.items() if freq/largest > idf_treshold)
    return result

def filter_unknown_vocabs(json_data, vocabs, stopwords=set(), seq_len_treshold=1):
    def check_seq(seq):
        allow_term = vocabs.__contains__ if vocabs is not None else lambda x: True
        events = []
        to_remove = False
        for x in seq['sequence']:
            # This branch is neede because if we remove a term, we must remove
            # the consecutive $BRANCH token if it exists
            if to_remove:
                to_remove = False
                if x['call'] == '$BRANCH':
                    continue
            call = x['call']
            to_remove = not allow_term(call) or call in stopwords
            if not to_remove:
                events.append(x)

        seq['sequence'] = events
        return len(events) >= seq_len_treshold

    for pkg in sal.get_packages(doc=json_data):
        pkg['data'] = list(filter(check_seq, pkg['data']))

def parse_word_list(fname):
    with open(fname) as fp:
        for word in fp:
            word = word.strip().split("#", 1)[0]
            if word != "":
                yield word

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='The input JSON file.')
    parser.add_argument('outfile', nargs='?', default=None, help='The output file. Default: standard-output')
    parser.add_argument('--min-len', default=3, type=int, help='The minimum call-sequence length accepted. Default: %(default)r')
    parser.add_argument('--idf-treshold', default=.0025, type=float, help='Any call whose IDF is below this value will be ignored. Default: %(default)r')
    parser.add_argument('--list-tf', action="store_true", help='Instead of filtering, show the term frequency of the input file.')
    parser.add_argument('--skip-filter-low', dest="run_tf", action="store_false", help='By default filters low-frequency terms; this disables this filter.')
    parser.add_argument('--stopwords', help='Provide a file (one term per line) with terms that must be removed from any sequence.')
    get_nprocs = common.parser_add_parallelism(parser)
    args = parser.parse_args()

    try:
        with common.smart_open(args.infile) as f:
            data = json.load(f)
        if args.list_tf or args.run_tf:
            tf = get_term_frequency(data, nprocs=get_nprocs(args), seq_len_treshold=args.min_len)
        else:
            tf = None
        if args.list_tf:
            out = open(args.outfile, 'w') if args.outfile is not None else sys.stdout
            for term, freq in sorted(tf.items(), key=lambda x:(x[1],x[0]), reverse=True):
                print(freq, term, file=out)
        else:
            if tf is not None:
                vocabs = get_common_vocabs(tf, idf_treshold=args.idf_treshold)
            else:
                vocabs = None

            if args.stopwords is not None:
                stopwords = set(parse_word_list(args.stopwords))
            else:
                stopwords = set()

            filter_unknown_vocabs(data, vocabs, stopwords, seq_len_treshold=args.min_len)
            if args.outfile is None:
                json.dump(data, sys.stdout)
            else:
                with common.smart_open(args.outfile, 'wt') as f:
                    json.dump(data, f)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == '__main__':
    main()
