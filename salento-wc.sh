#!/bin/bash
home=$(dirname "$0")
"$home"/salento-to-txt.py -f "$@" | sed -E -e 's/ /\n/g' | sort | uniq -c
