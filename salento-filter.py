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

def get_term_frequency(doc, nprocs, min_seq_len=1):
    result = State()
    with common.finish(concurrent.futures.ThreadPoolExecutor(max_workers=nprocs), accumulator=result) as executor:
        for pkg in sal.get_packages(doc=doc):
            for seq in sal.get_sequences(pkg=pkg):
                if len(sal.get_calls(seq=seq)) >= min_seq_len:
                    executor.submit(seq_term_frequency, seq)
    return result.get()

def get_common_vocabs(tf, idf_treshold=0.0025):
    if len(tf) == 0:
        return set()
    (_,largest), = tf.most_common(1)
    result = set(term for term, freq in tf.items() if freq/largest > idf_treshold)
    return result

def parse_word_list(fname):
    with open(fname) as fp:
        for word in fp:
            word = word.strip().split("#", 1)[0]
            if word != "":
                yield word

def filter_unknown_vocabs(js,
    vocabs=None,
    stopwords=set(),
    alias=dict(),
    min_seq_len=3,
    branch_tokens=set(['$BRANCH'])):
    """
    By default sequences with 2 or fewer are filtered out.

        >>> small_seq = Sequence([Call('foo'), Call('bar')])
        >>> pkg = Package([small_seq], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js)
        >>> len(pkg)
        0

    If we change the set the minimum size to 0, we do not filter based on lenght:

        >>> small_seq = Sequence([Call('foo'), Call('bar')])
        >>> pkg = Package([small_seq], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, min_seq_len=0)
        >>> len(pkg)
        1

    We can use stop words to eliminate calls, in this case by removing
    the call 'foo' we actually remove the first sequence (as it falls below
    the acceptable minimum length):

        >>> seq1 = Sequence([Call('foo'), Call('bar')])
        >>> seq2 = Sequence([Call('foo'), Call('bar'), Call('baz')])
        >>> pkg = Package([seq1, seq2], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, stopwords=['bar'], min_seq_len=2)
        >>> len(pkg)
        1
        >>> len(pkg[0])
        2
        >>> pkg[0][0].call == 'foo' and pkg[0][1].call == 'baz'
        True

    We can vocabs to limit the accepted terms, in this case by removing
    the call 'bar' (note that we are not filtering out based on minimum length):

        >>> seq1 = Sequence([Call('foo'), Call('bar')])
        >>> seq2 = Sequence([Call('foo'), Call('bar'), Call('baz')])
        >>> pkg = Package([seq1, seq2], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, vocabs=['foo', 'baz'], min_seq_len=0)
        >>> len(pkg)
        2
        >>> len(pkg[0])
        1
        >>> pkg[0][0].call
        'foo'
        >>> len(pkg[1])
        2
        >>> pkg[1][0].call, pkg[1][1].call
        ('foo', 'baz')

    By default there's a notion of a branch token; when a non-branch token is
    removed (because it is a stop word or because it is not in the vocabs),
    all succeeding branch tokens are removed. In the following example we have
    two branch tokens that are removed because 'foo' is removed.

        >>> seq1 = Sequence([Call('foo'), Call('X'), Call('Y'), Call('bar')])
        >>> pkg = Package([seq1], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js, stopwords=['foo'], min_seq_len=0, branch_tokens=('X','Y'))
        >>> len(pkg)
        1
        >>> list(pkg[0].terms)
        ['bar']

    We can supply a map of aliases; the terms are replaced before filtering.

        >>> seq1 = Sequence([Call('baz'), Call('X'), Call('Y'), Call('bar')])
        >>> pkg = Package([seq1], name='p').js
        >>> pkg = VPackage(pkg)
        >>> filter_unknown_vocabs(pkg.js,
        ...     stopwords=['foo'],
        ...     min_seq_len=0,
        ...     branch_tokens=('X','Y'),
        ...     alias={'baz': 'foo', 'bar': 'ZZZ'})
        >>> len(pkg)
        1
        >>> len(pkg[0])
        1
        >>> list(pkg[0].terms)
        ['ZZZ']
    """
    ds = sal.Dataset.from_js(js, lazy=True)
    if alias is not None and len(alias) > 0:
        ds.translate_calls(alias)
    ds.filter_calls(vocabs=vocabs, stopwords=stopwords, branch_tokens=branch_tokens)
    ds.filter_sequences(min_length=min_seq_len)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='The input JSON file.')
    parser.add_argument('outfile', nargs='?', default=None, help='The output file. Default: standard-output')
    parser.add_argument('--min-len', default=3, type=int, help='The minimum call-sequence length accepted. Default: %(default)r')
    parser.add_argument('--idf-treshold', default=.25, type=float, help='A percentage. Any call whose IDF is below this value will be ignored. Default: %(default).2f%%')
    parser.add_argument('--stop-words-file', help='Provide a file (one term per line) with terms that must be removed from any sequence. Practically, this step removes terms from the vocabulary.')
    parser.add_argument('--alias-file', help='Provide a YAML file with the alias replacing each term that matches a key per value.')
    get_nprocs = common.parser_add_parallelism(parser)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--skip-filter-low', dest="run_tf", action="store_false", help='Disables the low-frequency filter.')
    group.add_argument('--vocabs-file', help='Disables the low-frequency filter. Uses the supplied vocabolary file, filtering any term that is not in the vocabulary.')

    args = parser.parse_args()

    try:
        with common.smart_open(args.infile, 'rt') as f:
            data = json.load(f)

        if args.run_tf:
            tf = get_term_frequency(data, nprocs=get_nprocs(args), min_seq_len=args.min_len)
            vocabs = get_common_vocabs(tf, idf_treshold=(args.idf_treshold / 100))
        elif args.vocabs is not None:
            vocabs = set(parse_word_list(args.vocab))
        else:
            vocabs = None

        if args.alias_file is not None:
            import yaml
            alias = yaml.load(open(args.alias_file))
        else:
            alias = dict()

        if args.stop_words_file is not None:
            stopwords = set(parse_word_list(args.stop_words_file))
        else:
            stopwords = set()

        filter_unknown_vocabs(data,
            vocabs=vocabs,
            stopwords=stopwords,
            alias=alias,
            min_seq_len=args.min_len
        )
        if args.outfile is None:
            json.dump(data, sys.stdout)
        else:
            with common.smart_open(args.outfile, 'wt') as f:
                json.dump(data, f)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == '__main__':
    main()
