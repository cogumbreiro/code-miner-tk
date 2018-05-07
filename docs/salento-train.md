# Salento training

## 1-step tutorial

`salento-train.py` trains an API-Usage Model given a Salento Call-Sequence
Dataset.

**Input:** `dataset.tar.bz2`

**Outputs:**
1. `dataset-clean.tar.bz2`: a smaller version of your dataset
   which has fewer/smaller traces by removing really uncommon terms. See `salento-filter.py` for more information.
2. `save/`: the Tensorflow API-Usage Model.
3. `save.tar.bz2`: the compressed Tensorflow API-Usage Model
4. `train.log`: a log-file of training the API-Usage Model

**Note**: In directory `save` there will many temporary files that can be
*safely* removed after training.

**Example:**
```
$ ls                 # Ensure that we have our input dataset
dataset.tar.bz2
$ ./salento-train.py # Train our API-Usage Model
...

$ ls                 # We now have the generated API-Usage Model
dataset.tar.bz2 dataset-clean.tar.bz2 save.tar.bz2 save train.log
```

# Tips

`salento-train.py` has many parameters that can be queried with the `--help` parameter.
```
$ salento-train.py --help
usage: salento-train.py [-h] [-C DIRNAME] [-i INFILE] [-f ARGS_FILE]
                        [--print-args] [--save-dir SAVE_DIR]
                        [--log-file LOG_FILE] [--config-file CONFIG_FILE]
                        [--backup-file BACKUP_FILE]
                        [--stop-words-file STOP_WORDS_FILE]
                        [--idf-treshold IDF_TRESHOLD] [--dry-run]
                        [--skip-clean-data] [--skip-log] [--skip-backup]
                        [--echo] [--salento-home SALENTO_HOME]
                        [--python-bin PYTHON_BIN]

Trains a Salento API-Usage model.
...
```

You can change the working directory with `-C`; all the arguments are relative to this parameter:
```
$ ls output/
dataset.tar.bz2
$ salento-train.py -C output
...
```
We can override any default argument by creating a `train.yaml` file.
To generate the defaults for a `train.yaml` file you can run the following
command:
```
$ ./salento-train.py --print-args > output/train.yaml
```


