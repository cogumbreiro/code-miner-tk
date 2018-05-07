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
import shutil

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

# https://stackoverflow.com/a/3041990
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def remove_temp(ctx):
    # By default we do NOT remove '{log_file}' because the user usually
    # wants to analyse the log file to ensure the model was correctly trained.
    common.delete_file(ctx.get_path('{infile_clean}'))
    # remove directory and its contents
    try:
        shutil.rmtree(ctx.get_path('{save_dir}'))
    except FileNotFoundError:
        pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trains a Salento API-Usage model.")
    parser.add_argument("-C", dest="dirname", default=".", help="Change the work directory. Default: %(default)r")
    parser.add_argument("-i", dest="infile", default="dataset.json.bz2", help="The Salento Packages JSON dataset. Default: %(default)r")
    parser.add_argument("-f", dest="args_file", default="train.yaml", help="Pass command-line options via an YAML configuration file.")
    parser.add_argument("--print-args", action="store_true", help="Print the arguments as an YAML configuration file and exit.")
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
    parser.add_argument("--keep-temp", dest="clean_temp", action="store_false", help="Keeps temporary files when successful.")
    parser.add_argument("--rm-all", action="store_true", help="Delete all generated files (asks before deleting) and exits.")
    parser.add_argument("--rm-temp", action="store_true", help="Delete all temporary files (asks before deleting) and exits.")

    args = parser.parse_args()
    cwd = os.getcwd()
    prev_dir = args.dirname
    os.chdir(args.dirname)
    if os.path.exists(args.args_file):
        import yaml
        ns = argparse.Namespace()
        ns.__dict__ = yaml.load(open(args.args_file))
        # rewind the working dir as it could have been overriden
        args = parser.parse_args(namespace=ns)
        if prev_dir != args.dirname:
            os.chdir(cwd)
            os.chdir(args.dirname)
    if args.print_args:
        import yaml
        yaml.dump(args.__dict__, stream=sys.stdout, default_flow_style=False)
        sys.exit(0)

    args.infile_clean = common.split_exts(args.infile)[0] + "-clean.json.bz2"

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
        if args.rm_all or args.rm_temp:
            infiles = ['{infile_clean}', '{log_file}', '{save_dir}']
            if args.rm_all:
                infiles.append('{backup_file}')
            
            infiles = ", ".join(map(repr, map(ctx.get_path, infiles)))
            if query_yes_no("Remove " + infiles + "?"):
                remove_temp(ctx)
                common.delete_file(ctx.get_path('{log_file}'))
                if args.rm_all:
                    common.delete_file(ctx.get_path('{backup_file}'))
                sys.exit(0)
            else:
                # Signal error when user changes their mind
                sys.exit(1)
        try:
            M.make(ctx, args, target="{backup_file}")
            # After building everything, we try to clean temporary files
            # if needed
            if args.clean_temp:
                remove_temp(ctx)

        except ValueError as e:
            print("ERROR:", e, file=sys.stderr)
            sys.exit(1)


    except KeyboardInterrupt:
        sys.exit(1)
    
if __name__ == '__main__':
    main()
 
