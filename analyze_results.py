"""Consolidated report over whatever results have been collected so far in
results/ci_results.json (from run_experiment.py) and
results/ci_category_results.json (from run_category_experiment.py).

Safe to run after any subset of sizes/trials has been collected -- sections
with too little data to be informative are skipped with a note rather than
crashing, so this can be re-run incrementally as a sweep builds up.
"""
import json
import math
import os

HERE = os.path.dirname(__file__)
CI_RESULTS = os.path.join(HERE, "results", "ci_results.json")
CI_CATEGORY_RESULTS = os.path.join(HERE, "results", "ci_category_results.json")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def report_part_a(ci):
    print("=" * 78)
    print("PART A: Critical-failure rate P(crown misses MSCC for both letters) vs n")
    print("=" * 78)
    rows = ci.get("part_a", [])
    if not rows:
        print("(no Part A results yet -- run: python run_experiment.py A <n> <trials>)")
        return
    print(f"{'n':>7} {'trials':>7} {'events':>7} {'rate':>10} {'95% CI':>22} {'n*rate':>8}")
    xs, ys = [], []
    for r in sorted(rows, key=lambda r: r["n"]):
        n, trials, ev, rate, lo, hi = r["n"], r["trials"], r["critical_failures"], r["rate"], r["ci_lo"], r["ci_hi"]
        print(f"{n:7d} {trials:7d} {ev:7d} {rate:10.5f} [{lo:.5f}, {hi:.5f}]   {n*rate:8.3f}")
        if ev > 0:
            xs.append(math.log(n))
            ys.append(math.log(rate))

    nptx = len(xs)
    if nptx < 3:
        print(f"\n(need >=3 sizes with at least one observed failure for the log-log "
              f"regression; have {nptx} -- collect more trials, especially at larger n, "
              f"to enable it)")
        return
    mean_x, mean_y = sum(xs) / nptx, sum(ys) / nptx
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        print("\n(all sizes with failures are identical -- cannot fit a regression)")
        return
    slope = cov / var
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    s2 = sum(r ** 2 for r in resid) / (nptx - 2)
    se_slope = math.sqrt(s2 / var)
    print(f"\nLog-log regression (points with >=1 event, n={nptx}):")
    print(f"  rate ~ n^k with k = {slope:.3f} +/- {1.96*se_slope:.3f} (95% CI)")
    print(f"  (O(1/n) predicts k = -1.0; k=0 -- i.e. constant rate -- would be firmly "
          f"excluded if CI excludes 0)")


def report_part_b(ci):
    print()
    print("=" * 78)
    print("PART B: Correctness of the optimistic verdict vs ground truth")
    print("=" * 78)
    rows = ci.get("part_b", [])
    if not rows:
        print("(no Part B results yet -- run: python run_experiment.py B <n> <trials>)")
        return
    print(f"{'n':>7} {'trials':>7} {'wrong':>7} {'95% CI upper bound':>22}")
    total_trials = total_wrong = 0
    for r in sorted(rows, key=lambda r: r["n"]):
        print(f"{r['n']:7d} {r['trials']:7d} {r['wrong']:7d} {r['ci_hi']:22.5f}")
        total_trials += r["trials"]
        total_wrong += r["wrong"]

    if total_trials == 0:
        return
    from scipy.stats import beta as beta_dist
    lo = 0.0 if total_wrong == 0 else beta_dist.ppf(0.025, total_wrong, total_trials - total_wrong + 1)
    hi = 1.0 if total_wrong == total_trials else beta_dist.ppf(0.975, total_wrong + 1, total_trials - total_wrong)
    print(f"\nPooled across all Part B sizes: {total_wrong}/{total_trials} wrong verdicts")
    print(f"95% CI on P(wrong): [{lo:.6f}, {hi:.6f}]")


def report_categories(cat):
    print()
    print("=" * 78)
    print("Category-level rates across whatever sizes were collected")
    print("=" * 78)
    if not cat:
        print("(no category results yet -- run: python run_category_experiment.py <n> <trials>)")
        return
    sizes = sorted(int(n) for n in cat.keys())
    if len(sizes) < 2:
        print(f"(only one size ({sizes[0] if sizes else '-'}) collected -- need at least two "
              f"to compare scaling; run python run_category_experiment.py <n> <trials> at a "
              f"second size)")
        if sizes:
            n = sizes[0]
            print(f"\nRates at n={n}:")
            for cname, r in sorted(cat[str(n)]["categories"].items(), key=lambda kv: -kv[1]["rate"]):
                print(f"  {cname:30s} {r['fail']:4d}/{r['total']:4d}  rate={r['rate']:.4f}  "
                      f"95% CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
        return

    n_lo, n_hi = sizes[0], sizes[-1]
    pred_ratio = math.sqrt(n_hi / n_lo)
    cats_lo = cat[str(n_lo)]["categories"]
    cats_hi = cat[str(n_hi)]["categories"]
    common = sorted(set(cats_lo) & set(cats_hi))
    if not common:
        print(f"(no check names in common between n={n_lo} and n={n_hi})")
        return
    print(f"Comparing n={n_lo} vs n={n_hi} (predicted ratio under Theta(n^-1/2) scaling: "
          f"{pred_ratio:.2f})\n")
    for cname in common:
        r_lo, r_hi = cats_lo[cname], cats_hi[cname]
        ratio = r_lo["rate"] / r_hi["rate"] if r_hi["rate"] > 0 else float("inf")
        print(f"{cname:28s}  n={n_lo}: {r_lo['rate']:.4f} [{r_lo['ci_lo']:.4f},{r_lo['ci_hi']:.4f}]   "
              f"n={n_hi}: {r_hi['rate']:.4f} [{r_hi['ci_lo']:.4f},{r_hi['ci_hi']:.4f}]   "
              f"ratio={ratio:.2f}")


if __name__ == "__main__":
    ci = load_json(CI_RESULTS) or {"part_a": [], "part_b": []}
    cat = load_json(CI_CATEGORY_RESULTS) or {}
    report_part_a(ci)
    report_part_b(ci)
    report_categories(cat)
