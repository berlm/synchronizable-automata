import sys
import json
import os
import re
import time
from collections import defaultdict
from scipy.stats import beta as beta_dist
from sync_algo import generate_random_automaton, run_modified_algorithm

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results", "ci_category_results.json")


def clopper_pearson(successes, trials, alpha=0.05):
    if trials == 0:
        return (0.0, 1.0)
    lower = 0.0 if successes == 0 else beta_dist.ppf(alpha / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta_dist.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return (float(lower), float(upper))


def category(name):
    name = re.sub(r'_cycle_\d+', '', name)
    name = re.sub(r'(_\d+)+$', '', name)
    return name


def load():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save(data):
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def run_level(n, trials, seed_offset=500000):
    t0 = time.time()
    cat_fail = defaultdict(int)
    cat_total = defaultdict(int)
    n_critical = 0
    for t in range(trials):
        nn, letters = generate_random_automaton(n, k=2, seed=seed_offset + t)
        res = run_modified_algorithm(nn, letters, compute_ground_truth=False)
        if res.critical_failure:
            n_critical += 1
            for name, failed in res.failures.items():
                if not name.startswith("cycle_compat") and not name.startswith("const_stable") \
                   and not name.startswith("stable_pair") and not name.startswith("gcd_colouring") \
                   and not name.startswith("big_cluster"):
                    cat_total[category(name)] += 1
                    if failed:
                        cat_fail[category(name)] += 1
            continue
        cats_this_run = defaultdict(bool)
        for name, failed in res.failures.items():
            cat = category(name)
            cats_this_run[cat] = cats_this_run[cat] or failed
        for cat in set(category(name) for name in res.failures):
            cat_total[cat] += 1
            if cats_this_run[cat]:
                cat_fail[cat] += 1

    result = {}
    for cat in cat_total:
        fail, tot = cat_fail[cat], cat_total[cat]
        lo, hi = clopper_pearson(fail, tot)
        result[cat] = {"fail": fail, "total": tot, "rate": fail / tot, "ci_lo": lo, "ci_hi": hi}
    dt = time.time() - t0
    print(f"n={n} trials={trials} critical={n_critical}  took={dt:.1f}s")
    for cat, r in sorted(result.items(), key=lambda kv: -kv[1]["rate"]):
        print(f"  {cat:30s} {r['fail']:4d}/{r['total']:4d}  rate={r['rate']:.4f}  "
              f"95% CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")

    data = load()
    data[str(n)] = {"trials": trials, "critical": n_critical, "categories": result}
    save(data)


if __name__ == "__main__":
    n, trials = int(sys.argv[1]), int(sys.argv[2])
    run_level(n, trials)
