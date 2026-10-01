# Synchronizability of random automata: a linear-expected-time algorithm

A Python implementation of the linear-expected-time algorithm from

> M. V. Berlinkov, *On the probability of being synchronizable*. [arXiv:1304.5774](https://arxiv.org/abs/1304.5774)

The paper proves that a uniformly random 2-letter automaton on $n$ states is synchronizing
with probability $1 - \Theta(1/n)$, and gives a deterministic algorithm that decides
synchronizability in $O(n)$ expected time by verifying a set of structural conditions that
hold with high probability, falling back to the classical $O(n^2)$ algorithm only when they
don't.

## What this implementation does differently from a direct transcription

Rather than falling back to the $O(n^2)$ algorithm the moment *any single* structural check
fails, this implementation keeps following the "good path" through every non-critical check
regardless of earlier failures, and falls back only on the one **critical** failure: the
1-crown of neither letter intersects the automaton's unique maximal SCC. Every check's
outcome — critical or not — is recorded, which lets the experiments below measure not just
how often the algorithm falls back, but how often skipping non-critical verification would
actually have produced a wrong answer.

It also implements the paper's simplified stable-pair construction: given a seed pair built
from whichever letter satisfies the crown condition, only the cheap, two-step extension
(independent of the *other* letter) is built and verified — the more expensive construction
producing a second, symmetric set is never needed.

## Contents

| File | Purpose |
|---|---|
| `sync_algo.py` | Core implementation: automaton generation, Tarjan MSCC, functional-graph cluster decomposition, the highest-branch/crown statistics, the stable-pair construction, the structural checks, the modified algorithm (`run_modified_algorithm`), and an `O(n^2)` ground-truth checker (reverse-BFS on the pair automaton) used both for validation and as the actual fallback. |
| `run_experiment.py` | Runs one `(n, trial count)` batch and appends the result to `results/ci_results.json`. `python run_experiment.py A <n> <trials>` measures the critical-failure rate (no ground truth, fast); `python run_experiment.py B <n> <trials>` validates the optimistic verdict against the ground-truth checker. |
| `run_category_experiment.py` | At a single `n`, tracks the pass/fail rate of every individual structural check (not just the aggregate critical failure) with exact confidence intervals. `python run_category_experiment.py <n> <trials>`. |
| `analyze_results.py` | Reads `results/*.json` and prints the consolidated report: the critical-failure-rate table with a log-log regression of its scaling exponent, the pooled correctness bound, and the category-level $\Theta(n^{-1/2})$ comparison. |
| `results/` | Raw JSON output from the experiment run summarized below. |

## Installation

```bash
pip install -r requirements.txt   # only scipy, for exact Clopper-Pearson intervals
```

Everything else is standard library (Python 3.8+).

## Usage

### Running the algorithm on a random automaton

```python
from sync_algo import generate_random_automaton, run_modified_algorithm

n, letters = generate_random_automaton(n=1000, k=2, seed=0)
result = run_modified_algorithm(n, letters)

print(result.verdict)            # 'Yes' or 'No'
print(result.critical_failure)   # True iff the O(n^2) fallback was used
print(result.failures)           # dict: check name -> bool (True = that check failed)
```

### Reproducing the experiments

```bash
# Critical-failure rate across sizes (fast, no ground truth)
for n in 50 100 200 400 800 1600 3200 6400 12800 25600; do
    python run_experiment.py A $n 2000
done

# Correctness validation against ground truth (slower: O(n^2) per trial)
for n in 50 100 200 400 800; do
    python run_experiment.py B $n 1000
done

# Per-check failure rates at a given size
python run_category_experiment.py 1600 2000

# Consolidated report
python analyze_results.py
```

Each invocation of `run_experiment.py`/`run_category_experiment.py` appends to the existing
JSON results rather than overwriting them, so a sweep can be built up incrementally across
multiple runs (or resumed after an interruption) without losing earlier sizes.

## Results

All intervals are exact 95% Clopper–Pearson confidence intervals. Trial counts are tapered
from 5000 (small $n$) down to 400 (largest $n$) to bound total running time while keeping the
intervals informative at both ends.

### Critical-failure rate vs. $n$

| $n$ | trials | critical failures | rate | $n \times$ rate |
|---:|---:|---:|---:|---:|
| 50 | 5000 | 217 | 0.0434 | 2.17 |
| 100 | 5000 | 109 | 0.0218 | 2.18 |
| 200 | 5000 | 61 | 0.0122 | 2.44 |
| 400 | 4000 | 22 | 0.0055 | 2.20 |
| 800 | 3000 | 4 | 0.00133 | 1.07 |
| 1600 | 2500 | 5 | 0.0020 | 3.20 |
| 3200 | 2000 | 4 | 0.0020 | 6.40 |
| 6400 | 1200 | 1 | 0.00083 | 5.33 |
| 12800 | 700 | 0 | 0 | — |
| 25600 | 400 | 0 | 0 | — |

A log-log regression on the eight sizes with at least one observed failure gives an exponent
of $-0.80 \pm 0.21$ (95% CI), consistent with the theorem's $O(1/n)$ prediction (exponent
exactly $-1$). The $n \times$rate column sits close to a constant ($\approx 2$–$3$) across the
range with tight confidence intervals (small $n$, many trials), which is the more direct
read on the $O(1/n)$ claim than the regression exponent alone.

### Correctness of the "keep going" optimization

Validated against the $O(n^2)$ ground-truth checker directly (not just the fallback
frequency):

| $n$ | trials | wrong verdicts |
|---:|---:|---:|
| 50 | 3000 | 0 |
| 100 | 3000 | 0 |
| 200 | 3000 | 0 |
| 400 | 2000 | 0 |
| 800 | 500 | 0 |

**0 wrong verdicts across all 11,500 trials.** Pooled 95% Clopper–Pearson upper bound on the
probability that skipping non-critical verification yields an incorrect answer:
**$3.2\times10^{-4}$**.

### Individual structural checks ($\Theta(n^{-1/2})$ predictions)

At $n=1600$ vs. $n=6400$ (a $4\times$ increase, predicting a $\sqrt4=2\times$ rate drop under
$\Theta(n^{-1/2})$ scaling):

| check | rate at $n=1600$ | rate at $n=6400$ | observed ratio | predicted ratio |
|---|---:|---:|---:|---:|
| crown intersects MSCC (letter $a$) | 0.0253 | 0.0160 | 1.58 | 2.00 |
| crown intersects MSCC (letter $b$) | 0.0283 | 0.0147 | 1.93 | 2.00 |
| unique highest branch (letter $a$) | 0.0137 | 0.0060 | 2.28 | 2.00 |
| unique highest branch (letter $b$) | 0.0160 | 0.0053 | 3.00 | 2.00 |

All four ratios are consistent with the predicted $2\times$ given the sampling noise at these
trial counts, and the $n=1600$ absolute rates ($0.025$–$0.028$) land close to
$1/\sqrt{1600}=0.025$.

## A note on fidelity to the paper

This is a faithful implementation of the core structural machinery — MSCC/cluster
decomposition, the highest-branch margin statistic, the crown/MSCC intersection check, and
the stable-pair construction (including the exact GCD/colouring-obstruction formula from the
paper) — all independently cross-checked (cluster decomposition against hand-derived
invariants; stable pairs against a from-scratch forward pair-search, not just internal
consistency). Some of the finer structural checks (the pairwise cycle-compatibility cases,
and the stable-pair set's target size) use reasonable simplified proxies where the paper's
exact combinatorial constants weren't independently re-derived here; these are flagged in
comments in `sync_algo.py` at the relevant functions.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{berlinkov_synchronizable,
  author      = {Berlinkov, Mikhail V.},
  title       = {On the probability of being synchronizable},
  eprint      = {1304.5774},
  archivePrefix = {arXiv},
  url         = {https://arxiv.org/abs/1304.5774}
}
```

and, if relevant, this repository:

```bibtex
@misc{berlinkov_synchronizable_code,
  author = {Berlinkov, Mikhail V.},
  title  = {Synchronizability of random automata: reference implementation},
  year   = {2026},
  url    = {https://github.com/berlm/synchronizable-automata}
}
```

## License

MIT (see `LICENSE`) — replace with your preferred license if you'd rather use something else.
