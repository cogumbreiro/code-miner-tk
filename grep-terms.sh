#!/bin/bash
grep -r --include='*.wc' "$1" "$2" -m 1 | cut -d: -f1 | sed 's/.wc$//'
