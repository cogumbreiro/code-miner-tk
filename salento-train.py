#!/usr/bin/env python3
import os
import os.path
import sys
import shlex
import json
import tarfile
import string
import time
import subprocess
import glob

if __name__ == '__main__':
    # Ensure we load our code
    CODE_MINER_HOME = os.path.abspath(os.path.dirname(sys.argv[0]))
    sys.path.insert(0, os.path.join(CODE_MINER_HOME, "src"))

import common
import make

################################################################################

M = make.Makefile()

@M.rule(source="{infile}", target="{infile_clean}")
def clean_data(ctx, args):
    cmd = [
        os.path.join(CODE_MINER_HOME, 'salento-filter.py'),
        ctx.get_path('{infile}'),
        ctx.get_path('{infile_clean}'),
        '--idf-treshold',
        str(args.idf_treshold),
    ]
    stop_words = ctx.get_path('{stop_words_file}')
    if os.path.exists(stop_words):
        cmd.append('--stop-words-file')
        cmd.append(stop_words)

    if args.echo:
        print(" ".join(map(shlex.quote, cmd)))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("ERROR cleaning", file=sys.stderr)
        raise KeyboardInterrupt


def train(ctx, args):
    save_dir = ctx.get_path("{save_dir}")
    # 1. Get script path
    cmd = [
        args.python_bin,
        os.path.join(args.salento_home, "src/main/python/salento/models/low_level_evidences/train.py"),
        ctx.get_path("{infile_clean}" if args.clean_data else "{infile}"),
        '--save',
        save_dir,
    ]


    # 3. Get configuration file
    config = ctx.get_path("{config_file}")

    if os.path.exists(config):
        cmd.append("--config")
        cmd.append(config)

    if args.log:
        log_file = ctx.get_path('{log_file}')
        stdout = open(log_file, "w")
    else:
        stdout = None

    if args.echo:
        print(" ".join(map(shlex.quote, cmd)))
    result = subprocess.run(cmd, stdout=stdout, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print("ERROR training", file=sys.stderr)
        if args.log:
            print(open(log_file).read(), file=sys.stderr)
        raise KeyboardInterrupt

def parse_checkpoint_file(fname):
    with open(fname) as fp:
        for line in fp:
            line = line.strip()
            key, val = line.split(': ')
            # We use basename because we do not want hardcoded paths
            yield (key, os.path.basename(json.loads(val)))

def get_save_dir_files(dirname):
    files = ["model.pbtxt", "config.json", "model.pb", "checkpoint"]
    files = set(os.path.join(dirname, x) for x in files)
    checkpoint = set(dict(parse_checkpoint_file(os.path.join(dirname, 'checkpoint'))).values())
    for x in checkpoint:
        x = os.path.join(dirname, x)
        if os.path.exists(x):
            files.add(x)
        found = set(glob.glob(x + ".*"))
        files = files.union(found)
    return files

def backup_files(target_filename, dirname):
    files = get_save_dir_files(dirname)
    tf = tarfile.open(target_filename, "w")
    for fname in files:
        tf.add(fname)
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

def normalize_path(path):
    in_path = path
    path = os.path.abspath(path)
    prefix = os.path.abspath(os.getcwd())
    path = path[len(prefix) + 1:] if path.startswith(prefix) else path
    return path 

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Runs the Salento trainer.")
    parser.add_argument("-C", dest="dirname", default=".", help="Change the work directory. Default: %(default)r")
    parser.add_argument("-i", dest="infile", default="dataset.json.bz2", help="The Salento Packages JSON dataset. Default: %(default)r")
    parser.add_argument("--save-dir", default="save", help="The default Tensorflow model directory. Default: %(default)r")
    parser.add_argument("--log-file", default="train.log", help="Log filename; path relative to directory name unless absolute path. Default: %(default)r")
    parser.add_argument("--config-file", default="config.json", help="Configuration filename; path relative to directory name unless absolute path. Default: Salento's configuration.")
    parser.add_argument("--backup-file", default="save.tar.bz2", help="Backup save dir archive name. Default: %(default)r")
    # For cleaning the dataset
    parser.add_argument("--stop-words-file", default="stop-words.txt", help="The stop-words to filter out (only given if the file exists).")
    parser.add_argument('--idf-treshold', default=.25, type=float, help='A percentage floating point number. Any call whose IDF is below this value will be ignored. Default: %(default).2f%%')

    parser.add_argument("--dry-run", action="store_true", help="Do not actually run any program, just print the commands.")
    parser.add_argument("--skip-clean-data", dest="clean_data", action="store_false", help="Do not clean the data.")
    parser.add_argument("--skip-log", dest='log', action="store_false", help="Skip logging.")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backing up the save directory.")
    parser.add_argument("--echo", action="store_true", help="Print out commands that it is running.")
    common.parser_add_salento_home(parser, dest="salento_home")
    parser.add_argument("--python-bin", default="python3", help="Python3 binary. Default: %(default)r")
    args = parser.parse_args()
    args.infile_clean = common.split_exts(args.infile)[0] + "-clean.json.bz2"
    os.chdir(args.dirname)
    if args.clean_data:
        source = "{infile_clean}"
    else:
        source = "{infile}"
    
    M.rule(source=source,
    targets=[
        "{save_dir}/model.pbtxt",
        "{save_dir}/config.json",
        "{save_dir}/model.pb",
        "{save_dir}/checkpoint"
    ])(train) # register train in `M`

    try:
        ctx = make.FileCtx(make.EnvResolver(vars(args), normalize_path))
        try:
            M.make(ctx, args, target="{backup_file}")
        except ValueError as e:
            print("ERROR:", e, file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(1)
    
if __name__ == '__main__':
    main()
 
