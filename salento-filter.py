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
import argparse
import numpy as np
import random
import os
import pickle
import json

def filter_unknown_vocabs(json_data, vocabs):
    def check_seq(seq):
        events = seq['sequence']
        events = list(filter(lambda x: x['call'] in vocabs, events))
        seq['sequence'] = events
        return len(events) > 0 and (len(events) > 1 or events[0]['call'] != "TERMINAL")

    for pkg in json_data['packages']:
        pkg['data'] = list(filter(check_seq, pkg['data']))

def parse_word_list(fname):
    with open(fname) as fp:
        for word in fp:
            word = word.strip()
            if word != "":
                yield word

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('infile', help='input JSON file')
    argparser.add_argument('outfile', help='output JSON file')
    argparser.add_argument('--whitelist', help="""An end-ofline-separated file \
with the accepted vocabulary; any term outside of this vocabolary will be \
ignored.""")
    args = argparser.parse_args()

    with common.smart_open(args.infile) as f:
        data = json.load(f)

    if args.whitelist is not None:
        vocabs = set(parse_word_list(args.whitelist))
        filter_unknown_vocabs(data, vocabs)

    with common.smart_open(args.outfile, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
