#!/usr/bin/env python3
#!/usr/bin/env python3
import sys
import os
import os.path
import warnings
import cmd
import argparse
import collections
import numpy as np
from operator import *
from typing import *

# Shut up Tensorflow
if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

from argparse import Namespace
from replui import argparse_cmd, parse_ranges, repl_format, REPLExit
from salui import make_app, ASequence, load_state_anomaly
from collections import Counter
from statedist import CallDistNorm, TermDistNorm

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input data file')
    parser.add_argument('out', help='output data file')
    parser.add_argument('--dirname', '-d', default="save",
                        help='directory to load model from')
    args = parser.parse_args()

    with make_app(args.filename, args.dirname) as app:
        app.init()
        seqs = (seq for pkg in app.pkgs for seq in pkg)
        db = load_state_anomaly(seqs,
            state_count=3,
            threshold=0.2,
            #accept_state=lambda x: x.name=="1"
        )
        with open(args.out, 'w') as fp:
            json.dump(db, fp)

if __name__ == '__main__':
    main()

