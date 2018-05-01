# Prologue

A language model gives us the probability of the last call in a given sequence
of calls. Let `L(s)` be the probability of the last-call of an API
call-sequence for a certain language model `L`.

The probability `P(s)` of a certain API call sequence be the commulative
language probability of all of its sub-sequences:

    P(s) = \mul_i P(s_i)

For instance, the probability of call sequence ˋc1 c2 c3ˋ is given by:

    P(c1 c2 c3) = L(c1) * L(c1 c2) * L(c1 c2 c3)

The *log-likelihood* anomaly score is defined as

    ll(s) = -log(P(s))

The KLD divergence takes a set of sequences `\vec s` and yields the mean
log likelihood of a set of sequences.

    log(1/n) + ( \sum ll(s_i) ) / n

where `n` is the number of sequences given in `\vec s`. Since 
`n` is small, we can simplify the expression to:

```
( \sum ll(s_i) ) / n
```

Further, to simplify the computation of our expression, we do:

```
-log (\prod P(s_i)) / n 
```

# Limitations of Salento 1.0

In this section we identify the limitations of Salento 1.0 and for each
of which we propose a recommendation.
Salento 1.0 uses KLD to detect anomalies. It groups the input sequences by
last-location and computes the KLD score of each group.

We can summarize the limitations of Salento 1.0 as follows:

1. Only function-exit points are considered
2. KLD bias
3. Unknown behaviour is considered anomalous
4. Low next-call probability does not imply anomaly

**1. Only function-exit points are considered.** Salento 1.0 will group
API-sequence calls by last location and assign a KLD divergence score to each
location. This is means that, in practice, Salento 1.0 will only report errors
at a couple of exit points per function, ignoring all other locations.
Furthermore, it will not report the actual location of the error, just where
the sequences incidentally terminate.

This leads to a poor UX experience, as the user has no way of knowing why a
certain location is anomalous, since the anomaly be in a different function.

*Recommendation:* we must range over all locations of the dataset.

**2. KLD bias.** The KLD score suffers from two limitations:

- outlier sequences are hidden in large groups
- the score grows with the length of the sequence calls

The KLD score averages the log-likelihoods of a set of sequences.
This means that *it is difficult to identifying an anomalous sequence in a set
of many non-anomalous sequences*.

*Recommendation:* in group-based scores we want
to single out outliers, by taking the most anomalous, rather than
computing average anomaly score.

Because the KLD score grows with the length of each sequence, Salento 1.0
will mistake long but not anomalous sequences for anomalous. Such pattern
shows up when there is high code code reuse, and in loops/branches, as these
yield multiple call sequences. This also means that functions with a sizeable
amount of code will be considered anomalous.

*Recommendation:* one idea is to analyze sub-sequences rather than a whole
sequence to identify which *portion* of the execution is anomalous.

**3. Unknown behaviour considered anomalous.** In a case where the API usage
pattern is *unknown* most, if not all, calls of a given sequence will have a
near-zero probability. The discussed KLD scoring algorithm will treat these
sequences as highly anomalous, which is unintersting to the end user --- it
might only be interesting to the engineer responsible for training the
API-usage model.

*Recommendation:* introduce new  techniques that disregard unknown behaviours.

**4. Low next-call probability does not imply anomaly.** 
The KLD diveregence score only considers the probability of each call in a
sequence, ignoring what the probabilities of other calls at a given point are.
For some sequence, we may have a call with a low probability and any other call
in that context would also yeild a low probability. The API usage pattern simply 
might just not know what to do next. The anomaly should be relative to other
calls in that context, and not just the probability of the next call.

*Recommendation:* consider the probability distribution of other calls
in the anomaly score.

# New metric: max-min likelihood

Let `P(c|s)` be the probability of the next call `c` given a sequence `s`.
Let the probability of max-call, notation `max(P(c'|s))`, be the maximum
probability of any call `c'` in the language vocabolary.
The **maximum likelihood**, notation `ml`, is defined as the ratio between the 
probabilities of the next call and the max call:
```
ml(c|s) = P(c|s) / max(P(c'|s))
```
A max-likelihood score ranges from 0 to 1, where 0 is unlikely and 1 is
likely. The maximum likelihood addresses problem (4), by taking into
consideration the calling context.


Let the min-call probability be the point in a call-sequence which exibits
the lowest max-likeilihood, which is given by iterating over all sub-sequences
of a sequence `s` that start at point 0 and computing the max-likelihood.

The max-min-likelihood identifies sequence of calls that have an average
high likelihood *and*, at the same time, have a few outliers, which
we obtain by computing the geometric mean of the max-likelihood.


```
mml(s) = (ml(s) ^ 2 + (1 - min-call(s)) ^ 2) / 2
```


*Problem (2.1): outlier sequences are hidden in large groups.*

*Problem (2.2): the score grows with the length of the sequence calls.*
We argue that  `mml` is more resilient to the commulative effect of long
sequences. The max-likelihood hides outliers in long sequences, because it
uses the arithmetic mean of each max-call probabilities. However, given
we also take into account the smallest max-call likelihood, we are still
able to single out these outliers.

*Problem (3): unknown behaviour considered anomalous.* Given that `mml`
favours a high average max-likelihood, unknown behaviours are ignored.

*Problem (4): low next-call probability does not imply anomaly.* We use
the max-call likelihood exactly to counter this problem.

# Salento 2.0

Salento 2.0 is our new version that targets the limitations of Salento 1.0.
To address problems 2--4 we introduce the max-likelihood score. To address
problem 1: our algorithm ranges over all sub-sequences that start at call 0.

Additionally, we introduce sequence-based metrics, rather than just ranking
a set of sequences. In Salento 2.0 there are two kinds of metrics: high-level
metrics which group sequences per source code locations; low-level metrics
which identify anomalies per sequence. The former is more useful to track down
more "obvious" bugs, those of which can be understood without knowning the
runtime state. The latter is necessary to understand anomalous behaviour and
helps the user understand how to reach a certain anomalous state at runtime.
