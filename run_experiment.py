import sys
import json
import os
import time
from collections import defaultdict
from scipy.stats import beta as beta_dist
from sync_algo import generate_random_automaton, run_modified_algorithm

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "ci_results.json")


def clopper_pearson(successes, trials, alpha=0.05):
    if trials == 0:
        return (0.0, 1.0)
    lower = 0.0 if successes == 0 else beta_dist.ppf(alpha / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta_dist.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (float(lower), float(upper))


def load():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"part_a": [], "part_b": []}


def save(data):
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def run_part_a_level(n, trials, seed_offset=0):
    t0 = time.time()
    n_critical = 0
    for t in range(trials):
        nn, letters = generate_random_automaton(n, k=2, seed=seed_offset + t)
        res = run_modified_algorithm(nn, letters, compute_ground_truth=False)
        if res.critical_failure:
            n_critical += 1
    lo, hi = clopper_pearson(n_critical, trials)
    row = {"n": n, "trials": trials, "critical_failures": n_critical,
           "rate": n_critical / trials, "ci_lo": lo, "ci_hi": hi}
    dt = time.time() - t0
    print(f"[A] n={n:6d} trials={trials:5d} critical={n_critical:4d} "
          f"rate={n_critical/trials:.5f}  95% CI=[{lo:.5f}, {hi:.5f}]  "
          f"n*rate={n*n_critical/trials:.3f}  took={dt:.1f}s")
    data = load()
    data["part_a"] = [r for r in data["part_a"] if r["n"] != n] + [row]
    data["part_a"].sort(key=lambda r: r["n"])
    save(data)


def run_part_b_level(n, trials, seed_offset=100000):
    t0 = time.time()
    n_wrong = 0
    n_checked = 0
    for t in range(trials):
        nn, letters = generate_random_automaton(n, k=2, seed=seed_offset + t)
        res = run_modified_algorithm(nn, letters, compute_ground_truth=True)
        if res.ground_truth is not None:
            n_checked += 1
            if res.correct is False:
                n_wrong += 1
    lo, hi = clopper_pearson(n_wrong, n_checked)
    row = {"n": n, "trials": n_checked, "wrong": n_wrong,
           "rate": n_wrong / max(n_checked, 1), "ci_lo": lo, "ci_hi": hi}
    dt = time.time() - t0
    print(f"[B] n={n:6d} trials={n_checked:5d} wrong={n_wrong:4d}  "
          f"95% CI=[{lo:.5f}, {hi:.5f}]  took={dt:.1f}s")
    data = load()
    data["part_b"] = [r for r in data["part_b"] if r["n"] != n] + [row]
    data["part_b"].sort(key=lambda r: r["n"])
    save(data)


if __name__ == "__main__":
    part, n, trials = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if part == "A":
        run_part_a_level(n, trials)
    elif part == "B":
        run_part_b_level(n, trials)
    else:
        raise ValueError(part)
