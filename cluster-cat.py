#!/usr/bin/env python3

import sys
import json

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prints all entries of all cluster, one entry per line.")
    args = parser.parse_args()
    for line in sys.stdin:
        for elem in json.loads(line):
            print(elem)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # User break, it's ok to quit
        pass
