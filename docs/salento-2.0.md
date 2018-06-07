# Prologue

A language model gives us the probability of the last term in a given
sequence of terms. Let $`L(t|s)`$ be the probability of the last term
$`t`$ given that the system has executed $`s`$ for a certain language
model $`L`$.

The probability $`P(s)`$ of a certain API call sequence
$`s = t_0 ... t_i`$ be the comulative language probability
of all of its sub-sequences $`t_0 ... t_i`$:

```math
    P(s) = \prod_i^n L(t_i | t_0 ... t_{i - 1})\quad \text{where } s = t_0 ... t_n
```

For instance, the probability of sequence $`t_1 t_2 t_3`$ is given by:

```math
    P(t_0 t_1 t_2) = L(t_0 |) \times L(t_1 | t_0) \times L(t_2 | t_0 t_1)
```

The *log-likelihood* anomaly score is defined as
```math
    \mathrm{ll}(s) = -\log(\mathrm{P}(s))
```
The KLD divergence takes a set of sequences $`\vec s`$ and yields the mean
log likelihood of a set of sequences.
```math
    \log(\frac{1}{n}) + \frac{\sum \mathrm{ll}(s_i)}{n}
```
where $`n`$ is the number of sequences given in $`\vec s`$. Since
$`n`$ is small, we can simplify the expression to:

```math
\frac{\sum{\mathrm{ll}(s_i)}}{n}
```

Further, to simplify the computation of our expression, we do:

```math
-\frac{\log (\prod P(s_i))} n
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

# Normalized likelihood

**Max-term probability.** Let the max-term probability of sequence `s`,
notation $`\mathrm{max}\ \mathrm{P}(s)`$, be the maximum probability of
any term $`t`$ in the language vocabulary such that
```math
\forall t, \mathrm{max\ P}(s) \ge \mathrm{P}(t | s)
```
 and there exists a term $`t_\mathrm{max}`$ such that
$`\mathrm{max\ P}(s) = \mathrm{P}(t_\mathrm{max} | s)`$

The **normalized likelihood**, notation $`nl(t|s)`$, is defined as the
ratio between the probabilities of the next term and the max term:
```math
\mathrm{nl}(c|s) = \frac{
    \mathrm{P}(c|s)
    }{
    \mathrm{max\ P}(s)
    }
```

A normalized-likelihood score ranges from 0 to 1, where 0 is unlikely and 1 is
likely. The maximum likelihood addresses problem (4), by taking into
consideration the calling context.

## Implementation details

In the implementation terms are divided into two classes. A term
can be either a call or a state. Each API call has some contextual information,
defined as a sequence of states. The distinction plays no role in the language
model. However, such a distinction is crucial when computing the probability
of the max term: given some distribution probability $`P(s)`$ the implementation
has to restrict the domain (*ie*, the range of terms) to distinguish between
the terms that are calls and the terms that are states.

In short, the implementation includes two variations (restrictions) of
$`\mathrm{max\ P}(s)`$, one for calls $`\mathrm{maxCall\ P}(s)`$,
and one for states $`\mathrm{maxState}_i\ \mathrm{P}(s)`$.

> **Call domain.** A term is considered to be a call if, and only if,
> its name does not start with a number followed by `#`. **TODO:** Call
> names that include the string `#` are therefore problematic and are
> currently **ignored** (or considered to be a state-token).

> **State domain.** States are ordered, so it is required to know the
> position~$`i`$ in which the state appears. Note that the end of the
> state is also recorded, so a state with $`n`$ states will have $`n+1`$
> possible distribution probabilities.
> A term is considered to be a state in the $`i`$-th position if, and only
> if, it is the sentinel term (which in the implementation is term
> `END_MARKER`) or the term starts with number $`i`$ followed by `#`.


# New metric: dip anomaly

The dip-anomaly metric takes a sequence of *likelihoods* (from 0 to 1) and
returns an *anomaly* score (from 0 to 1). Identifies as anomalous sequences
with a high average likelihood *and*, at the same time, includes a few outliers
(elements with low likelihood).

```math
\mathrm{dip}(s) = \frac{\mathrm{avg}(s) ^ 2 + (1 - \min(s)) ^ 2} 2
```

or in Numpy terms:

```python
dip = lambda x : (x.mean() ** 2 + (1 - x.min()) ** 2) / 2
```

> **Why do we use the geometric mean?** We want to penalize heavily when
>  one of the two components does not match our goals; that is 
>
> We also want to highlight that this metric favours a **single** anomaly;
> as we are taking the smallest anomaly and when we have multiple anomalies,
> we are simply lowering the average.


**Note 1:** to convert from an anomaly score to a likelihood score, you
can do `1 - anomaly`.

**Note 2:** We use this anomaly metric by giving it the normalized likelihood
of each call in a sequence:
```math
\mathrm{nl}(t_0|), ..., \mathrm{nl}(t_n + 1 | t_0 ... t_n+1)
```

## Examples

The base case is a sequence where we have one highly likely call and one highly
unlikely:
```
>>> dip(np.array([1.0, 0]))
0.625
```
Since the anomaly score ranges from 0 to 1 (where higher is more anomalous),
this sequence is not very anomalous.

As we can see below, as the sequence grows, the the anomaly increases slowly
(because we square the average):
```
>>> dip(np.array([0]))
0.5
>>> dip(np.array([1]))
0.5
>>> dip(np.array([1.0, 0]))
0.625
>>> dip(np.array([1.0, 1.0, 0]))
0.7222222222222222
>>> dip(np.array([1.0, 1.0, 1.0, 0]))
0.78125
>>> dip(np.array([1.0, 1.0, 1.0, 1.0, 0]))
0.8200000000000001
```



*Problem (2.1): outlier sequences are hidden in large groups.*
We do *not* accumulate scores of various sequences; instead we always pick
the most anomalous sequence for a given location.

*Problem (2.2): the score grows with the length of the sequence calls.*
Our metric favours longer sequences with fewer anomalies.

*Problem (3): unknown behaviour considered anomalous.* Given that `dip`
favours a high average likelihood, unknown behaviours are ignored.

*Problem (4): low next-call probability does not imply anomaly.* We propose
the normalized likelihood to counter this problem.

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
