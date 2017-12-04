#!/bin/bash
home=$(dirname "$0")
"$home"/salento-to-txt.py --eol '' -f "$@" | sed -E -e 's/ /\n/g' | sed '/^\s*$/d' | sort | uniq -c
