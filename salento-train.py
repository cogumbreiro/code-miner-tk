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
import make
import os.path
import shlex
import json
import tarfile
import string
import time
import subprocess
from functools import reduce

################################################################################

M = make.Makefile()

@M.rule(source="{infile}",
targets=[
    "{save_dir}/model.pbtxt",
    "{save_dir}/config.json",
    "{save_dir}/model.pb",
    "{save_dir}/checkpoint"
])
def train(ctx, args):
    save_dir = ctx.get_path("{save_dir}")
    # 1. Get script path
    cmd = [
        args.python_bin,
        os.path.join(args.salento_home, "src/main/python/salento/models/low_level_evidences/train.py"),
        ctx.get_path("{infile}"),
    ]


    # 3. Get configuration file
    config = ctx.get_path("{config_file}")

    if args.resume:
        cmd.append("--continue_from")
        cmd.append(save_dir)
    else:
        cmd.append("--save")
        cmd.append(save_dir)
        if os.path.exists(config):
            cmd.append("--config")
            cmd.append(config)
    if not args.skip_log:
        log_file = ctx.get_path('{log_file}')
        stdout = open(log_file, "w")
    else:
        stdout = None

    if args.echo:
        print(" ".join(map(shlex.quote, cmd)))
    result = subprocess.run(cmd, stdout=stdout, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print("ERROR training", file=sys.stderr)
        raise KeyboardInterrupt

def parse_checkpoint_file(fname):
    with open(fname) as fp:
        for line in fp:
            line = line.strip()
            key, val = line.split(': ')
            yield (key, json.loads(val))

def get_save_dir_files(dirname):
    return ["model.pbtxt", "config.json", "model.pb", "checkpoint"] + \
        list(dict(parse_checkpoint_file(os.path.join(dirname, 'checkpoint'))).values())

def backup_files(target_filename, dirname):
    files = get_save_dir_files(dirname)
    tf = tarfile.open(target_filename, "w")
    for fname in files:
        tf.add(os.path.join(dirname, fname), arcname=fname)
    tf.close()
    

@M.rule(sources=[
    "{save_dir}/model.pbtxt",
    "{save_dir}/config.json",
    "{save_dir}/model.pb",
    "{save_dir}/checkpoint",
], target="{backup_file}")
def backup(ctx, args):
    if args.skip_backup:
        return
    save_dir = ctx.get_path("{save_dir}")
    backup_file = ctx.get_path("{backup_file}")
    if args.dry_run:
        print("BACKUP: " + repr(save_dir))
        return
    try:
        backup_files(target_filename=backup_file, dirname=save_dir)
    except OSError as err:
        print("Error writing backup file %r:" % backup_file, err, file=sys.stderr)
        common.delete_file(backup_file)
        raise KeyboardInterrupt()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Runs the Salento trainer.")
    parser.add_argument("-C", dest="dirname", default=".", help="The target work directory. Default: %(default)r")
    parser.add_argument("-i", dest="infile", default="dataset.json.bz2", help="The Salento Packages JSON dataset. Default: %(default)r")
    parser.add_argument("--save-dir", default="save", help="The default Tensorflow model directory. Default: %(default)r")
    parser.add_argument("--log-file", default="train.log", help="Log filename; path relative to directory name unless absolute path. Default: %(default)r")
    parser.add_argument("--config-file", default="config.json", help="Configuration filename; path relative to directory name unless absolute path. Default: Salento's configuration.")
    parser.add_argument("--backup-file", default="save.tar.bz2", help="Backup save dir archive name. Default: %(default)r")

    parser.add_argument("--resume", action="store_true", help="Do not actually run any program, just print the commands.")
    parser.add_argument("--dry-run", action="store_true", help="Do not actually run any program, just print the commands.")
    parser.add_argument("--skip-log", action="store_true", help="Skip logging.")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backing up the save directory.")
    parser.add_argument("--echo", action="store_true", help="Print out commands that it is running.")
    common.parser_add_salento_home(parser, dest="salento_home")
    parser.add_argument("--python-bin", default="python3", help="Python3 binary. Default: %(default)r")
    args = parser.parse_args()

    try:
        ctx = make.FileCtx(make.EnvResolver(args.dirname, vars(args)))
        try:
            if args.resume or args.skip_backup:
                M.run(ctx, [train], args, force=args.resume)
            else:
                M.make(ctx, args, target="{backup_file}")
        except ValueError as e:
            print("ERROR:", e, file=sys.stderr)
            sys.exit(1)
        #do_backup(ctx)
    except KeyboardInterrupt:
        sys.exit(1)
    
if __name__ == '__main__':
    main()
 
