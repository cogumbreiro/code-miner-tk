# Salento usage

1. `salento-repl.py pop.json` loads a dataset to memory.
2. `kld` will give you a bird's-eye of your dataset

    ```
    > kld --limit 3
    id: 0 location: macopix-1.7.4/as-out/pop.c.as.bz2 | pop.c:1012 score: 12.8
    id: 0 location: macopix-1.7.4/as-out/pop.c.as.bz2 | pop.c:690 score: 12.2
    id: 0 location: macopix-1.7.4/as-out/pop.c.as.bz2 | pop.c:1009 score: 12.0
    ```

    **Notes:**

    Argument `--limit 3` shows the top-3 with the most anomolous score.

3. `seq` lets you drill down by sequence

    ```
    > seq * * --end 'pop.c:1012$'
    id: 25 count: 19 last: pop.c:1012
    id: 27 count: 15 last: pop.c:1012
    id: 29 count: 11 last: pop.c:1012
    id: 31 count: 7 last: pop.c:1012
    id: 33 count: 20 last: pop.c:1012
    id: 35 count: 16 last: pop.c:1012
    id: 37 count: 12 last: pop.c:1012
    id: 39 count: 8 last: pop.c:1012
    ```

    **Notes:**

    Arguments `* *` match all package- and all sequence-identifiers, respectively.

    Argument `--end 'pop.c:1012$'` matches any sequence whose the last location ends with `pop.c:1012`.


4. The sequence-field `log` yields the log-score per-sequence:

    ```
    > seq * * --end 'pop.c:1012$'  -f 'id: {seq.sid} len: {seq.count} score: {seq.log:.1f}'
    id: 25 len: 19 score: 12.6
    id: 27 len: 15 score: 13.2
    id: 29 len: 11 score: 13.7
    id: 31 len: 7 score: 12.3
    id: 33 len: 20 score: 18.9
    id: 35 len: 16 score: 18.4
    id: 37 len: 12 score: 17.4
    id: 39 len: 8 score: 12.6
    ```

    **Notes:**

    Argument `-f 'id: {seq.sid} len: {seq.count} score: 
    {seq.log:.1f}'` changes the formating so that we get the log-score
    of each sequence.

5. We can also visualize all sequences in a query by generating Graphviz `*.gv` files:

    ```
    > seq * * --end 'pop.c:1012$'  -f 'id: {seq.sid} len: {seq.count} score: {seq.log:.1f}' --viz
    ```

    If we run in a terminal (in the same working directory):

    ```bash
    $ ls  *.gv
    0-25.gv  0-27.gv  0-29.gv  0-31.gv  0-33.gv  0-35.gv  0-37.gv  0-39.gv
    ```

    **Note:**
    Argument `--viz` iterates over each sequence and creates a Graphviz file.
    You can change the filename with `--viz-fmt`.

6. Visualize Graphviz to identify an anomaly.

    ```
    $ xdot 0-35.gv
    ```

    ![xdot screenshot](xdot-example.png)

    Colored edges represent the call sequence being rendered, these are
    annotated with the similarity between the given call and the most
    probable call.

    Black edges represent the most probable call and is annotated with
    the probability of that call.

    In this example, the tool is warning us about two anomalies:
      * *Missed checking the return value of an `sprintf` call.*
      * *After checking the return value of function `popWriteLine` the function should end.*
