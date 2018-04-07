#!/bin/bash
usage() {
    echo Usage $0 "<BOUND>" 2> /dev/null;
    exit 1
}

[ -z "$1" ] && usage
lbound=$1
awk '{ if ($1 >= '${lbound}') print $2}'
