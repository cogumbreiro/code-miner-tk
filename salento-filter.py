#!/usr/bin/env python3
import sys
import os
import numpy as np
import concurrent.futures
import sys
import argparse
import random
import os
import pickle
import json

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

import common
import sal


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
    parser.add_argument('--idf-treshold', default=.25, type=float, help='A percentage. Any call whose IDF is below this value will be ignored. Default: %(default).2f%%')
    parser.add_argument('--stop-words-file', help='Provide a file (one term per line) with terms that must be removed from any sequence. Practically, this step removes terms from the vocabulary.')
    get_nprocs = common.parser_add_parallelism(parser)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--skip-filter-low', dest="run_tf", action="store_false", help='Disables the low-frequency filter.')
    group.add_argument('--vocabs-file', help='Disables the low-frequency filter. Uses the supplied vocabolary file, filtering any term that is not in the vocabulary.')

    args = parser.parse_args()

    try:
        with common.smart_open(args.infile) as f:
            data = json.load(f)

        if args.run_tf:
            tf = get_term_frequency(data, nprocs=get_nprocs(args), min_seq_len=args.min_len)
            vocabs = get_common_vocabs(tf, idf_treshold=(args.idf_treshold / 100))
        elif args.vocabs is not None:
            vocabs = set(parse_word_list(args.vocab))
        else:
            vocabs = None

        if args.stop_words_file is not None:
            stopwords = set(parse_word_list(args.stop_words_file))
        else:
            stopwords = set()

        sal.filter_unknown_vocabs(data, vocabs, stopwords, seq_len_treshold=args.min_len)
        if args.outfile is None:
            json.dump(data, sys.stdout)
        else:
            with common.smart_open(args.outfile, 'wt') as f:
                json.dump(data, f)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == '__main__':
    main()
