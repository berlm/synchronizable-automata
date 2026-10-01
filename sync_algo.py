"""
Implementation of the linear-expected-time synchronizability algorithm from
"On the probability of being synchronizable for a random automaton" (algorithm section),
MODIFIED per the requested change:

  Instead of falling back to the O(n^2) algorithm the moment ANY structural check fails,
  we keep following the "good path" -- i.e. we continue running the remaining checks
  regardless of earlier (non-critical) failures -- and only trigger the O(n^2) fallback
  on a CRITICAL failure: the 1-crown of neither letter intersects the (unique) MSCC.

  Every check's failure is counted, separately tracking:
    - "original" failures: non-critical checks that failed but did not stop the algorithm
    - "critical" failures: the crown/MSCC-intersection check failing for both letters

At the end, if no critical failure occurred, the algorithm outputs 'Yes' (synchronizing)
even if some non-critical checks failed along the way -- this is no longer guaranteed
correct in the worst case, so we separately validate against a ground-truth quadratic
checker to measure how often this optimistic shortcut is actually wrong.

Also incorporates the simplified stable-pair construction (matching the revised
Theorem 12 / thm:big_sync_sets in the paper): given a seed pair built from letter x
(hence independent of the other letter x_bar), only the CHEAP, two-step set Z
independent of x_bar is built and used -- the more expensive three-step construction
producing a second set independent of x itself is never needed, since Theorem 6
(thm:big_sync_sets) only ever requires the "other letter"'s set, via the a-cycle-
reduction argument (or its mirror). This halves the connectivity/gcd/colouring work
done per run compared to the earlier version of this file.
"""

import random
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Automaton representation and generation
# ----------------------------------------------------------------------------

def generate_random_automaton(n, k=2, seed=None):
    """Each letter is an independent uniformly random function Q -> Q."""
    rng = random.Random(seed)
    letters = []
    for _ in range(k):
        letters.append([rng.randrange(n) for _ in range(n)])
    return n, letters


# ----------------------------------------------------------------------------
# Step 1: Tarjan SCC + MSCC (sink SCC(s) of the condensation) extraction
# ----------------------------------------------------------------------------

def tarjan_scc(n, letters):
    """Standard iterative Tarjan SCC on the underlying digraph (union of all letters).
    Returns: comp_id[state] -> component index, and list of components (each a list of states).
    """
    adj = [[] for _ in range(n)]
    for trans in letters:
        for q in range(n):
            adj[q].append(trans[q])

    index_counter = [0]
    stack = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlink = [0] * n
    comp_id = [-1] * n
    components = []

    for start in range(n):
        if indices[start] != -1:
            continue
        # iterative tarjan using explicit work stack: (node, child_iter_index)
        work = [(start, 0)]
        while work:
            v, i = work[-1]
            if i == 0:
                indices[v] = lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            while i < len(adj[v]):
                w = adj[v][i]
                i += 1
                if indices[w] == -1:
                    work[-1] = (v, i)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])
            if recurse:
                continue
            work[-1] = (v, i)
            if i >= len(adj[v]):
                work.pop()
                if work:
                    p = work[-1][0]
                    lowlink[p] = min(lowlink[p], lowlink[v])
                if lowlink[v] == indices[v]:
                    comp = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp_id[w] = len(components)
                        comp.append(w)
                        if w == v:
                            break
                    components.append(comp)
    return comp_id, components


def find_msccs(n, letters):
    """Minimal SCCs = sink SCCs in the condensation (no outgoing edge leaves the SCC)."""
    comp_id, components = tarjan_scc(n, letters)
    has_outgoing = [False] * len(components)
    for trans in letters:
        for q in range(n):
            if comp_id[trans[q]] != comp_id[q]:
                has_outgoing[comp_id[q]] = True
    mscc_indices = [i for i in range(len(components)) if not has_outgoing[i]]
    return [components[i] for i in mscc_indices]


# ----------------------------------------------------------------------------
# Step 2: functional-graph cluster decomposition for a single letter, restricted
# to a state subset B on which that letter is closed (e.g. the unique MSCC).
# Computes: cluster id (cycle representative), cycle length, root offset,
# tree root (specific cycle vertex the state's tree hangs from), height.
# ----------------------------------------------------------------------------

@dataclass
class ClusterStructure:
    cluster: dict       # state -> cluster representative (smallest-index cycle state)
    cl: dict             # state -> cycle length of its cluster
    root: dict           # state -> offset of its tree's cycle-root within the cycle (0..cl-1)
    tree: dict           # state -> the specific cycle state whose tree this state hangs from
    height: dict         # state -> BFS height from the cycle (0 for cycle states)
    cycle_states: dict   # cluster representative -> ordered list of cycle states (index = offset)
    n_clusters: int


def compute_cluster_structure(trans, B):
    """trans: full transition array (state -> state) for one letter.
    B: iterable of states on which `trans` is closed (trans[q] in B for all q in B).
    Runs in O(|B|) time via the standard rho-shape walk.
    """
    Bset = set(B)
    color = {q: 0 for q in Bset}  # 0=unvisited,1=on current walk,2=done
    cluster = {}
    cl = {}
    root = {}
    tree = {}
    height = {}
    cycle_states = {}
    n_clusters = 0

    for start in Bset:
        if color[start] != 0:
            continue
        path = []
        pos = {}
        v = start
        while color[v] == 0:
            color[v] = 1
            pos[v] = len(path)
            path.append(v)
            v = trans[v]
        if color[v] == 1:
            # found a fresh cycle starting at path[pos[v]]
            cyc_start_idx = pos[v]
            cyc = path[cyc_start_idx:]
            rep = min(cyc)
            n_clusters += 1
            # assign root offsets 0..cl-1 walking the cycle from `rep`
            rep_idx = cyc.index(rep)
            ordered = cyc[rep_idx:] + cyc[:rep_idx]
            cycle_states[rep] = ordered
            clen = len(ordered)
            for offset, s in enumerate(ordered):
                cluster[s] = rep
                cl[s] = clen
                root[s] = offset
                tree[s] = s
                height[s] = 0
                color[s] = 2
            # the prefix path[0:cyc_start_idx] is a tail leading INTO the cycle at path[cyc_start_idx]
            entry = path[cyc_start_idx]
            for i in range(cyc_start_idx - 1, -1, -1):
                s = path[i]
                h = cyc_start_idx - i  # distance to the cycle
                cluster[s] = rep
                cl[s] = clen
                root[s] = root[entry]
                tree[s] = entry
                height[s] = h
                color[s] = 2
        else:
            # v is already 'done' -- path[] is a tail leading into existing structure at v
            base_h = height[v]
            for i in range(len(path) - 1, -1, -1):
                s = path[i]
                h = base_h + (len(path) - i)
                cluster[s] = cluster[v]
                cl[s] = cl[v]
                root[s] = root[v]
                tree[s] = tree[v]
                height[s] = h
                color[s] = 2

    return ClusterStructure(cluster, cl, root, tree, height, cycle_states, n_clusters)


# ----------------------------------------------------------------------------
# Step 3: "1-branch" statistics -- highest branch, uniqueness/margin, and its root.
#
# A candidate 1-branch is either a whole tree (rooted at a cycle state, height 0)
# or a subtree rooted at a height-1 state. We take the "height" of a candidate to be
# the maximum absolute height reached within it. We report the overall tallest
# candidate, whether it is unique, its margin over the second-tallest, and its root.
# ----------------------------------------------------------------------------

def compute_reverse_children(trans, B):
    children = defaultdict(list)
    for q in B:
        children[trans[q]].append(q)
    return children


def highest_branch_stats(cs: ClusterStructure, trans, B):
    """Returns (margin, root_of_highest_branch, tree_root_of_highest_branch, deepest_leaf)
    or None if there are no non-cycle states at all (degenerate).

    NOTE (simplification): the paper's "1-branch" is a maximal subtree with root height
    <= 1; a naive implementation comparing both height-0 and height-1 rooted subtrees as
    separate candidates is unsound, since a height-1 subtree is always properly contained
    in its own tree's height-0 (whole-tree) candidate -- they are never independent
    competitors, so comparing them against each other spuriously forces near-certain ties.
    We instead use the unambiguous, well-defined statistic of comparing WHOLE TREES (one
    candidate per cycle state that has a nontrivial tree attached) -- the tallest tree
    versus the second-tallest, both rooted at height 0. This is a faithful analogue of
    the same margin/crown idea, differing from the paper's finer-grained 1-branch only in
    not additionally sub-dividing within the tallest tree.
    """
    if not cs.height:
        return None
    children = compute_reverse_children(trans, B)
    max_height_in_subtree = {}
    deepest_in_subtree = {}

    by_height_desc = sorted(cs.height.keys(), key=lambda s: -cs.height[s])
    for s in by_height_desc:
        m, leaf = cs.height[s], s
        for c in children[s]:
            if cs.height[c] != cs.height[s] + 1:
                continue  # never follow a cycle edge, only strict tree ascent
            if c in max_height_in_subtree and max_height_in_subtree[c] > m:
                m, leaf = max_height_in_subtree[c], deepest_in_subtree[c]
        max_height_in_subtree[s] = m
        deepest_in_subtree[s] = leaf

    candidates = [(max_height_in_subtree[s], s) for s, h in cs.height.items() if h == 0]
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    top_h, top_root = candidates[0]
    second_h = candidates[1][0] if len(candidates) > 1 else -1
    margin = top_h - second_h
    return margin, top_root, cs.tree[top_root], deepest_in_subtree[top_root]


# ----------------------------------------------------------------------------
# Step 4: the CRITICAL check -- does the 1-crown of a letter intersect the MSCC?
#
# In this single-MSCC-restricted view B, "intersects the MSCC" is automatic (the whole
# structure lives inside B). The real (paper) statement is about the crown, computed on
# the FULL n-state automaton, intersecting the separately-identified MSCC B. We therefore
# compute the highest-branch / crown structure on the FULL automaton (all n states) for
# each letter, and check whether any crown state lies in B.
# ----------------------------------------------------------------------------

def crown_states(cs: ClusterStructure, trans, n, branch_stats):
    """States at height >= (second_highest_height + 1) within the tallest branch."""
    if branch_stats is None:
        return set()
    margin, top_root, top_tree_root, deepest_leaf = branch_stats
    children = compute_reverse_children(trans, range(n))
    stack = [top_root]
    visited = {top_root}
    max_reach = cs.height[top_root]
    subtree = []
    while stack:
        u = stack.pop()
        subtree.append(u)
        max_reach = max(max_reach, cs.height[u])
        for c in children[u]:
            # only ascend strictly into the tree (never follow the cycle edge back)
            if c not in visited and cs.height[c] == cs.height[u] + 1:
                visited.add(c)
                stack.append(c)
    threshold = max_reach - margin + 1
    return {u for u in subtree if cs.height[u] >= threshold}


def crown_intersects_mscc(n, trans, B):
    """Compute the full-automaton cluster structure & highest branch for `trans`,
    then check whether its crown intersects B."""
    cs_full = compute_cluster_structure(trans, range(n))
    stats = highest_branch_stats(cs_full, trans, range(n))
    if stats is None:
        return False, cs_full, stats
    crown = crown_states(cs_full, trans, n, stats)
    return len(crown & set(B)) > 0, cs_full, stats


# ----------------------------------------------------------------------------
# Ground truth: quadratic synchronizability check via reverse-BFS on the pair
# automaton from the diagonal, exactly as described in the algorithm section.
# Also serves as the O(n^2) fallback algorithm itself.
# ----------------------------------------------------------------------------

def is_synchronizing_quadratic(n, letters):
    """A automaton is synchronizing iff every pair of states can reach the diagonal,
    i.e. iff reverse-BFS from the diagonal (under preimages of each letter) covers
    all pairs {p,q}, p<q."""
    preimage = [defaultdict(list) for _ in letters]
    for li, trans in enumerate(letters):
        for q in range(n):
            preimage[li][trans[q]].append(q)

    visited = [[False] * n for _ in range(n)]
    dq = deque()
    for q in range(n):
        visited[q][q] = True
    for p in range(n):
        for q in range(p + 1, n):
            pass
    # seed: all diagonal pairs are trivially "reached"; BFS backward to find all
    # pairs {p,q} such that some word maps p,q to the same state.
    frontier = deque((q, q) for q in range(n))
    seen_pairs = set((q, q) for q in range(n))
    while frontier:
        u, v = frontier.popleft()
        for li in range(len(letters)):
            for pu in preimage[li][u]:
                for pv in preimage[li][v]:
                    a_, b_ = (pu, pv) if pu <= pv else (pv, pu)
                    if (a_, b_) not in seen_pairs:
                        seen_pairs.add((a_, b_))
                        frontier.append((a_, b_))
    total_pairs = n * (n - 1) // 2 + n
    return len(seen_pairs) == total_pairs


# ----------------------------------------------------------------------------
# Step 5: stable-pair extension.
#
# A pair {p,q} is "stable" here if, under repeated application of a single letter x,
# p and q eventually collide (reach the same state) -- i.e. same cluster and cycle-
# compatible offsets. Starting from one known-stable pair (root of the highest branch,
# and the cycle predecessor of its tree's root -- stable by construction since one step
# of `a` sends both into the same tree), we extend to more stable pairs via preimages
# under BOTH letters (mirroring the ground-truth reverse-BFS, capped at n^0.45 pairs
# for linear-time approximate verification).
# ----------------------------------------------------------------------------

def chain_extend(letters, given_pairs, walk_letter_idx, target_count):
    """Shared machinery for Lemma const_stable and Lemma many_stable: given a list of
    pairs already known to be stable and independent of `walk_letter_idx`, walk forward
    by repeatedly applying that letter to build new pairs. Each new pair is automatically
    stable (the image of a stable pair is always stable) and, since it was built using
    fresh randomness from `walk_letter_idx`, is conditionally independent of the OTHER
    letter. On a collision (revisiting an already-seen state, matching the lemma's
    stopping rule) we move to the next given pair, exactly as in the many-pairs case of
    Lemma many_stable; a single given pair with no collision recovers Lemma const_stable.
    Never raises on shortfall -- returns whatever was successfully built."""
    trans = letters[walk_letter_idx]
    result = []
    seen_states = set()
    for (p, q) in given_pairs:
        seen_states.add(p)
        seen_states.add(q)
    for (p0, q0) in given_pairs:
        p, q = p0, q0
        while len(result) < target_count:
            p, q = trans[p], trans[q]
            if p == q or p in seen_states or q in seen_states:
                break  # collision: this pair's chain is exhausted, move to the next given pair
            seen_states.add(p)
            seen_states.add(q)
            result.append((p, q) if p < q else (q, p))
        if len(result) >= target_count:
            break
    return result


def build_Z(letters, seed_pair, x_idx, k=10, target=None):
    """Single-output construction matching the revised Theorem 12: given a seed pair
    built from letter `x_idx` alone (hence independent of the other letter), produce
    Z -- independent of the OTHER letter -- via the cheap two-step chain: walk via the
    other letter (Lemma const_stable), then walk via x_idx (Lemma many_stable). This is
    the only set needed; the more expensive three-step construction producing a set
    independent of x_idx itself is never used. Returns (Z, info) where info records the
    count actually reached at each stage for failure bookkeeping."""
    assert len(letters) == 2
    xbar_idx = 1 - x_idx
    if target is None:
        target = max(2, int(math.ceil(len(letters[0]) ** 0.45)))

    I = chain_extend(letters, [seed_pair], walk_letter_idx=xbar_idx, target_count=k)
    info = {"const_stable": len(I)}
    if not I:
        return [], info

    Z = chain_extend(letters, I, walk_letter_idx=x_idx, target_count=target)
    info["many_stable"] = len(Z)
    return Z, info


# ----------------------------------------------------------------------------
# Step 6: big-cluster connectivity via the extended stable-pair graph.
# ----------------------------------------------------------------------------

def big_clusters(cs: ClusterStructure, n, threshold_exp=0.95):
    threshold = n ** threshold_exp
    sizes = defaultdict(int)
    for s, rep in cs.cluster.items():
        sizes[rep] += 1
    return {rep for rep, sz in sizes.items() if sz > threshold}, sizes


def big_cluster_graph_connected(cs: ClusterStructure, S, Z):
    if len(S) <= 1:
        return True, None
    parent = {r: r for r in S}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    edges = []
    for (p, q) in Z:
        rp, rq = cs.cluster.get(p), cs.cluster.get(q)
        if rp in S and rq in S and rp != rq:
            union(rp, rq)
            edges.append((p, q, rp, rq))
    roots = {find(r) for r in S}
    return len(roots) == 1, edges


# ----------------------------------------------------------------------------
# Step 7: gcd / colouring obstruction check (Lemma_stable_cluster), using the
# exact formula given in the paper: g | (root(p)-root(q)) - (x_cp - x_cq) - (h(p)-h(q)).
# ----------------------------------------------------------------------------

def gcd_colouring_check(cs: ClusterStructure, S, spanning_edges, all_pairs):
    if not S:
        return True, 1
    d = 0
    for rep in S:
        d = math.gcd(d, cs.cl[rep])
    if d <= 1:
        return True, d

    # assign cluster offsets x[] via DFS over the spanning edges, root offset 0
    adj = defaultdict(list)
    for (p, q, rp, rq) in spanning_edges:
        adj[rp].append((rq, p, q))
        adj[rq].append((rp, q, p))
    x = {}
    start = next(iter(S))
    x[start] = 0
    stack = [start]
    while stack:
        u = stack.pop()
        for (v, pu, pv) in adj[u]:
            if v not in x:
                # solve x[v] mod d from: root(pu)-root(pv) - (x[u]-x[v]) - (h(pu)-h(pv)) == 0 (mod d)
                x[v] = (x[u] - cs.root[pu] + cs.root[pv] + cs.height[pu] - cs.height[pv]) % d
                stack.append(v)
    for rep in S:
        x.setdefault(rep, 0)

    g = d
    for (p, q) in all_pairs:
        rp, rq = cs.cluster.get(p), cs.cluster.get(q)
        if rp in S and rq in S:
            diff = (cs.root[p] - cs.root[q]) - (x[rp] - x[rq]) - (cs.height[p] - cs.height[q])
            g = math.gcd(g, diff)
    return g == 1, g


# ----------------------------------------------------------------------------
# Step 8: pairwise cycle-compatibility checks (simplified proxy for the paper's
# three-case analysis in Theorem big_sync_sets). We use T_b := states in "small"
# clusters of b (size <= n^0.95) as a practical stand-in for the paper's T-hat_b.
# ----------------------------------------------------------------------------

BETA = 8  # small constant threshold for "big" cycles in this simplified check

def small_cluster_states(cs: ClusterStructure, n, threshold_exp=0.95):
    threshold = n ** threshold_exp
    sizes = defaultdict(int)
    for s, rep in cs.cluster.items():
        sizes[rep] += 1
    return {s for s, rep in cs.cluster.items() if sizes[rep] <= threshold}


def cycle_compatibility_checks(cs_x: ClusterStructure, cs_xbar: ClusterStructure, n):
    """Returns list of (case_name, ok) results. `cs_x` supplies the cycle structure
    (every deadlock pair is reduced to x-cycles, matching letter x playing the role of
    'a' in the paper's case analysis); `cs_xbar` supplies the small-cluster exclusion
    set T_{xbar} those cycles must mostly avoid."""
    T_xbar = small_cluster_states(cs_xbar, n)
    results = []
    # gather distinct cycles of `x` (by representative) with their ordered cycle states
    cycles = list(cs_x.cycle_states.items())
    big_cycles = [(rep, states) for rep, states in cycles if len(states) >= 2 * BETA]

    # Case 1: one cycle of big size -- fails if >= half its states are in T_xbar
    for rep, states in big_cycles:
        cnt = sum(1 for s in states if s in T_xbar)
        ok = cnt < len(states) / 2
        results.append((f"case1_cycle_{rep}", ok))

    # Case 2: two cycles of big size -- cumulative-count style pairwise check
    for i in range(len(big_cycles)):
        for j in range(i + 1, len(big_cycles)):
            rep_p, states_p = big_cycles[i]
            rep_q, states_q = big_cycles[j]
            sp, sq = len(states_p), len(states_q)
            if sp > sq:
                (rep_p, states_p, sp), (rep_q, states_q, sq) = (rep_q, states_q, sq), (rep_p, states_p, sp)
            zp = sum(1 for s in states_p[:sp] if s in T_xbar)
            zq = sum(1 for s in states_q[:sp] if s in T_xbar)
            ok = (zp + zq) < sp
            results.append((f"case2_{rep_p}_{rep_q}", ok))

    # Case 3: small cycles -- constant-depth expansion check
    small_cycles = [(rep, states) for rep, states in cycles if len(states) < 2 * BETA]
    for i in range(len(small_cycles)):
        rep_i, states_i = small_cycles[i]
        for s in states_i:
            ok = s not in T_xbar
            results.append((f"case3_{rep_i}_{s}", ok))

    return results


# ----------------------------------------------------------------------------
# Seed stable pair for the highest-branch root: pick the cycle state q such that
# r and q are GUARANTEED to collide after height(r) steps of the letter (cycle
# arithmetic), which is a well-defined, checkable stand-in for the paper's
# {r, predecessor-of-tree-root} construction.
# ----------------------------------------------------------------------------

def seed_pair_for_branch(cs: ClusterStructure, r):
    tree_root = cs.tree[r]
    rep = cs.cluster[r]
    clen = cs.cl[r]
    target_root = (cs.root[tree_root] - cs.height[r]) % clen
    q = cs.cycle_states[rep][target_root]
    if q == r:
        return None
    return (r, q) if r < q else (q, r)


# ----------------------------------------------------------------------------
# The MODIFIED algorithm: run every "good path" check in sequence, counting every
# failure. Only the crown/MSCC-intersection check failing for BOTH letters is
# treated as critical (triggers the O(n^2) fallback); every other failure is
# recorded and the algorithm keeps going.
# ----------------------------------------------------------------------------

@dataclass
class RunResult:
    n: int
    verdict: str                 # 'Yes' (optimistic) or 'Yes'/'No' (from fallback)
    used_fallback: bool
    critical_failure: bool
    failures: dict = field(default_factory=dict)   # check name -> bool (True=failed)
    ground_truth: bool = None
    correct: bool = None


def run_modified_algorithm(n, letters, seed=None, compute_ground_truth=True):
    failures = {}

    def record(name, ok):
        failures[name] = (not ok)

    # --- Step 1: MSCC ---
    msccs = find_msccs(n, letters)
    if len(msccs) > 1:
        # definitive structural fact, not a probabilistic "failure" -- automaton
        # provably not synchronizing.
        gt = is_synchronizing_quadratic(n, letters) if compute_ground_truth else None
        return RunResult(n, 'No', used_fallback=False, critical_failure=False,
                          failures=failures, ground_truth=gt, correct=(gt is False))

    B = set(msccs[0])
    record("core_size_ge_0.75n", len(B) >= 0.75 * n)

    trans_a, trans_b = letters[0], letters[1]
    cs_a = compute_cluster_structure(trans_a, B)
    cs_b = compute_cluster_structure(trans_b, B)

    record("clusters_le_5lnn_a", cs_a.n_clusters <= 5 * math.log(max(n, 2)))
    record("clusters_le_5lnn_b", cs_b.n_clusters <= 5 * math.log(max(n, 2)))

    stats_a = highest_branch_stats(cs_a, trans_a, B)
    stats_b = highest_branch_stats(cs_b, trans_b, B)
    record("unique_highest_branch_a", stats_a is not None and stats_a[0] > 0)
    record("unique_highest_branch_b", stats_b is not None and stats_b[0] > 0)

    # --- CRITICAL CHECK: crown intersects the MSCC, for each letter ---
    crown_ok_a, cs_a_full, hstats_a_full = crown_intersects_mscc(n, trans_a, B)
    crown_ok_b, cs_b_full, hstats_b_full = crown_intersects_mscc(n, trans_b, B)
    record("crown_intersects_mscc_a", crown_ok_a)
    record("crown_intersects_mscc_b", crown_ok_b)

    critical_failure = (not crown_ok_a) and (not crown_ok_b)
    if critical_failure:
        gt = is_synchronizing_quadratic(n, letters) if compute_ground_truth else None
        return RunResult(n, ('Yes' if gt else 'No') if gt is not None else 'FALLBACK',
                          used_fallback=True, critical_failure=True,
                          failures=failures, ground_truth=gt, correct=True if gt is not None else None)

    # --- Stable-pair construction (Theorem 14 -> single cheap set via revised Theorem 12) ---
    # WLOG pick whichever letter satisfies crown-intersection to source the seed pair
    # (the algorithm text notes at least one always does, given we passed the critical
    # check above); prefer "a" if both do, matching the paper's WLOG. The seed is built
    # from letter x; we only ever need Z independent of the OTHER letter xbar (Theorem 6
    # only requires xbar's set, via the a-cycle-reduction argument or its mirror), so we
    # never build the second, more expensive set independent of x itself.
    if crown_ok_a:
        x_idx, x_letter, xbar_letter = 0, "a", "b"
        seed_cs, seed_stats, cs_xbar, trans_xbar = cs_a, stats_a, cs_b, trans_b
    else:
        x_idx, x_letter, xbar_letter = 1, "b", "a"
        seed_cs, seed_stats, cs_xbar, trans_xbar = cs_b, stats_b, cs_a, trans_a
    record("stable_pair_seed", seed_stats is not None)

    target = max(2, int(math.ceil(n ** 0.45)))
    if seed_stats is not None:
        leaf = seed_stats[3]
        seed = seed_pair_for_branch(seed_cs, leaf)
    else:
        seed = None

    if seed is None:
        record("stable_pair_seed_valid", False)
        Z = []
    else:
        Z, info = build_Z(letters, seed, x_idx, k=10, target=target)
        record("const_stable", info.get("const_stable", 0) >= 10)
        # per the requested flexibility: only the FINAL extension is graded against the
        # full target: any nonzero yield is used as-is, shortfall is just recorded.
        record(f"stable_pair_extension_{xbar_letter}", info.get("many_stable", 0) >= target)

    S, sizes = big_clusters(cs_xbar, n)
    connected, edges = big_cluster_graph_connected(cs_xbar, S, Z)
    record(f"big_cluster_connectivity_{xbar_letter}", connected if S else True)

    ok_gcd, g = gcd_colouring_check(cs_xbar, S, edges or [], Z)
    record(f"gcd_colouring_{xbar_letter}", ok_gcd)

    cs_x = cs_a if x_letter == "a" else cs_b
    cyc_results = cycle_compatibility_checks(cs_x, cs_xbar, n)
    n_cyc_fail = 0
    for name, ok in cyc_results:
        record(f"cycle_compat_{name}", ok)
        if not ok:
            n_cyc_fail += 1

    verdict = 'Yes'
    gt = is_synchronizing_quadratic(n, letters) if compute_ground_truth else None
    correct = (gt is True) if gt is not None else None
    return RunResult(n, verdict, used_fallback=False, critical_failure=False,
                      failures=failures, ground_truth=gt, correct=correct)

