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
import os.path
import shlex
import json

def get_path(args, attr, default=None):
    fname = getattr(args, attr)
    if fname is None:
        fname = default
    if fname is None or os.path.isabs(fname):
        return fname
    return os.path.join(args.dirname, fname)

def do_train(args):
    # 1. Get script path
    train_py = os.path.join(args.salento_home, "src/main/python/salento/models/low_level_evidences/train.py")
    cmd = "%s %s" % (shlex.quote(args.python_bin), shlex.quote(train_py))
    
    # 2. Get filename
    cmd += " %s " % shlex.quote(get_path(args, "infile"))

    # 3. Get configuration file
    config = get_path(args, "config_file")
    if config is not None and os.path.exists(config):
        cmd += " --config " + shlex.quote(config)

    # 4. Add logging
    if not args.skip_log:
        log_file = get_path(args, "log_file")
        cmd += " | tee " + shlex.quote(log_file)
    try:
        common.run(cmd, silent=args.quiet, echo=args.echo, dry_run=args.dry_run)
    except KeyboardInterrupt:
        sys.exit(1)

def parse_checkpoint_file(fname):
    with open(fname) as fp:
        for line in fp:
            line = line.strip()
            key, val = line.split(': ')
            yield (key, json.loads(val))

def get_save_dir_files(dirname):
    return ["model.pbtxt", "config.json", "model.pb", "checkpoint"] + \
        list(dict(parse_checkpoint_file(os.path.join(dirname, 'checkpoint'))).values())

def backup_save_dir(dirname, target_filename):
    tf = tarfile.open(target_filename, "w")
    for fname in get_save_dir_files(dirname):
        tf.add(os.path.join(dirname, fname), arcname=fname)
    tf.close()

def do_backup(args):
    if args.skip_backup:
        return
    save_dir = get_path(args, "save_dir")
    if args.dry_run:
        print("BACKUP: " + repr(save_dir))
        return
    backup_save_dir(args.backup_file, save_dir)

def do_prepare(args):
    save_dir = get_path(args, "save_dir")
    if args.dry_run:
        print("MKDIR: " + save_dir)
    else:
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            if not args.force:
                print("%r directory already exists; remove it before continuing." % save_dir, file=sys.stderr)
                sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Runs the Salento trainer.")
    parser.add_argument("-C", dest="dirname", default=".", help="The target work directory. Default: %(default)r")
    parser.add_argument("-i", dest="infile", default="dataset.json.bz2", help="The Salento Packages JSON dataset. Default: %(default)r")
    parser.add_argument("--save-dir", default="save", help="The default Tensorflow model directory. Default: %(default)r")
    parser.add_argument("--log-file", default="train.log", help="Log filename; path relative to directory name unless absolute path. Default: %(default)r")
    parser.add_argument("--config-file", default=None, help="Configuration filename; path relative to directory name unless absolute path. Default: Salento's configuration.")
    parser.add_argument("--backup-file", default="save.tar.bz2", help="Backup save dir archive name. Default: %(default)r")
    
    parser.add_argument("--dry-run", action="store_true", help="Do not actually run any program, just print the commands.")
    parser.add_argument("--skip-log", action="store_true", help="Skip logging.")
    parser.add_argument("--skip_backup", action="store_true", help="Skip backing up the save directory.")
    parser.add_argument("--force", action="store_true", help="Ignore if saves directory exists.")
    parser.add_argument("--echo", action="store_true", help="Print out commands that it is running.")
    parser.add_argument("--quiet", action="store_true", help="Print out the output of training.")
    common.parser_add_salento_home(parser, dest="salento_home")
    parser.add_argument("--python-bin", default="python3", help="Python3 binary. Default: %(default)r")
    args = parser.parse_args()

    do_prepare(args)    
    do_train(args)
    do_backup(args)
    
if __name__ == '__main__':
    main()
 

