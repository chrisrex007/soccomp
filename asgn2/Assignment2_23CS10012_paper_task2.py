"""Task 2: rank-one greedy, baselines, non-convex local search and witnesses."""

from __future__ import annotations
import argparse
from itertools import product
import time
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import spsolve
import networkx as nx
from Assignment2_23CS10012_greedy_task1 import (
    LOW,
    HIGH,
    ROOT,
    Solver,
    system,
    save_json,
    rss,
    rng,
    graph_matrix,
    load_inputs,
    shortlist,
    choose,
)


class State:
    """Mutable inverse and equilibrium. Every scoring operation is solve-free."""

    def __init__(self, P, a, s):
        self.P = P
        self.a = a.copy()
        self.s = s
        self.G = linalg.inv(system(P, a).toarray(), check_finite=True)
        self.z = self.G @ (a * s)
        self.p = P @ self.z
        self.c = self.G.sum(axis=0)
        # Sparse-dense product only during setup; do not retain a second dense matrix.
        self.diag = np.einsum("ij,ji->i", P.toarray(), self.G)
        columns = np.unique(np.linspace(0, len(a) - 1, min(8, len(a)), dtype=int))
        inverse_residual = (
            system(P, a) @ self.G[:, columns] - np.eye(len(a))[:, columns]
        )
        if np.max(abs(inverse_residual)) > 1e-10:
            raise ArithmeticError("Inverse column residual check failed")
        self.solver = Solver()
        self.verifications = 0
        self.verify()

    def scores(self, pool):
        pool = np.asarray(pool, dtype=int)
        delta = np.array([LOW, HIGH])[None, :] - self.a[pool, None]
        den = 1 + delta * self.diag[pool, None]
        if np.any(den <= 0):
            raise ArithmeticError("Nonpositive Sherman-Morrison denominator")
        gains = (
            delta * (self.s[pool] - self.p[pool])[:, None] * self.c[pool, None] / den
        )
        return float(self.z.sum()) + gains

    def accept(self, v, endpoint):
        delta = endpoint - self.a[v]
        g = self.G[:, v].copy()
        h = np.asarray(self.P.getrow(v) @ self.G).ravel()
        den = 1 + delta * h[v]
        if den <= 0:
            raise ArithmeticError("Nonpositive update denominator")
        beta = delta / den
        Pg = self.P @ g
        res = self.s[v] - self.p[v]
        cv = self.c[v]
        self.z += beta * res * g
        self.p += beta * res * Pg
        self.c -= beta * cv * h
        self.diag -= beta * Pg * h
        self.G -= beta * np.outer(g, h)
        self.a[v] = endpoint

    def verify(self):
        independent = self.solver.solve(self.P, self.a, self.s)
        self.verifications += 1
        error = float(np.max(abs(independent - self.z)))
        residual = float(np.max(abs(system(self.P, self.a) @ self.z - self.a * self.s)))
        if error > 2e-8 or residual > 1e-10:
            raise ArithmeticError(f"Rank update drift: {error}, {residual}")
        return dict(max_error=error, residual=residual)


class FixedBase:
    """Woodbury evaluations against an immutable inverse; no large copy per trial."""

    def __init__(self, state, pool=None):
        self.state = state
        self.lookup = None
        if pool is not None:
            pool = np.asarray(pool, dtype=int)
            self.lookup = {int(v): i for i, v in enumerate(pool)}
            self.H = np.asarray(state.P[pool, :] @ state.G[:, pool])

    def evaluate(self, indices, values, vector=False):
        state = self.state
        S = np.array(indices, dtype=int)
        if not len(S):
            return state.z.copy() if vector else float(state.z.sum())
        delta = np.asarray(values) - state.a[S]
        if self.lookup is None:
            H = np.asarray(state.P[S, :] @ state.G[:, S])
        else:
            idx = [self.lookup[int(v)] for v in S]
            H = self.H[np.ix_(idx, idx)]
        K = np.eye(len(S)) + delta[:, None] * H
        b = delta * (state.s[S] - state.p[S])
        w = linalg.solve(K, b, check_finite=False)
        if np.max(abs(K @ w - b)) > 1e-9:
            raise ArithmeticError("Small-system residual too large")
        if vector:
            return state.z + state.G[:, S] @ w
        return float(state.z.sum() + state.c[S] @ w)


def fast(P, a, s, ids, pool, k, direction=1, out=None):
    start = time.perf_counter()
    state = State(P, a, s)
    setup = time.perf_counter() - start
    initial = float(state.z.sum())
    selected = []
    rows = []
    for step in range(1, k + 1):
        remaining = [int(v) for v in pool if v not in selected]
        scores = state.scores(remaining)
        choices = [
            (v, e, float(scores[i, j]))
            for i, v in enumerate(remaining)
            for j, e in enumerate((LOW, HIGH))
        ]
        v, e, f = choose(choices, direction)
        state.accept(v, e)
        diagnostics = state.verify()
        selected.append(v)
        actual = float(state.z.sum())
        if abs(actual - f) > 1e-8:
            raise ArithmeticError("Candidate and accepted objective disagree")
        rows.append(
            dict(
                k=step,
                added=int(ids[v]),
                index=v,
                alpha=e,
                f=actual,
                gain=direction * (actual - (rows[-1]["f"] if rows else initial)),
                seconds=time.perf_counter() - start,
                setup_seconds=setup,
                peak_mib=rss(),
                verification_solves=state.verifications,
                **diagnostics,
            )
        )
        if out:
            save_json(out, dict(initial=initial, rows=rows, complete=step == k))
    return dict(initial=initial, rows=rows, complete=True)


def baseline(P, a, s, ids, degree, kind, k=30, direction=1, pool=None, out=None):
    start = time.perf_counter()
    pool = np.arange(len(a)) if pool is None else np.asarray(pool)
    values = s if direction == 1 else 1 - s
    if kind == "I":
        score = values
    elif kind == "II":
        neighbor_sum = degree * (P @ values)
        score = (
            np.divide(
                values,
                neighbor_sum,
                out=np.where(values > 0, np.inf, 0.0),
                where=neighbor_sum != 0,
            )
            * degree
            / degree.sum()
        )
    else:
        raise ValueError(kind)
    order = pool[np.lexsort((ids[pool], -score[pool]))]
    solver = Solver()
    initial = float(solver.solve(P, a, s).sum())
    current = a.copy()
    rows = []
    for step, v in enumerate(order[:k], 1):
        current[v] = HIGH
        z = solver.solve(P, current, s)
        f = float(z.sum())
        rows.append(
            dict(
                k=step,
                added=int(ids[v]),
                index=int(v),
                alpha=HIGH,
                f=f,
                gain=direction * (f - (rows[-1]["f"] if rows else initial)),
                seconds=time.perf_counter() - start,
                peak_mib=rss(),
                total_solves=solver.calls,
            )
        )
    result = dict(initial=initial, rows=rows, complete=True)
    if out:
        save_json(out, result)
    return result


def random_trials(
    P, a, s, ids, k=30, direction=1, pool=None, pool_code=0, trials=20, out=None
):
    start = time.perf_counter()
    state = State(P, a, s)
    setup = time.perf_counter() - start
    pool = np.arange(len(a)) if pool is None else np.array(pool)
    all_rows = []
    for trial in range(trials):
        tick = time.perf_counter()
        order = rng(1000 + 100 * pool_code + trial).permutation(pool)[:k]
        chosen = []
        endpoints = []
        rows = []
        base = FixedBase(state, order)
        for step, v in enumerate(order, 1):
            candidates = [
                (int(v), e, base.evaluate(chosen + [int(v)], endpoints + [e]))
                for e in (LOW, HIGH)
            ]
            _, e, f = choose(candidates, direction)
            chosen.append(int(v))
            endpoints.append(e)
            rows.append(
                dict(
                    k=step,
                    added=int(ids[v]),
                    index=int(v),
                    alpha=e,
                    f=f,
                    seconds=time.perf_counter() - tick,
                    peak_mib=rss(),
                )
            )
        verification = a.copy()
        verification[chosen] = endpoints
        if abs(Solver().solve(P, verification, s).sum() - rows[-1]["f"]) > 1e-7:
            raise ArithmeticError("Random verification failed")
        rows[-1]["seconds"] = time.perf_counter() - tick
        all_rows.append(rows)
        if out:
            save_json(
                out,
                dict(
                    initial=float(state.z.sum()),
                    setup_seconds=setup,
                    trials=all_rows,
                    complete=False,
                ),
            )
    rows = summarize_random_trials(all_rows, setup, k)
    result = dict(
        initial=float(state.z.sum()),
        setup_seconds=setup,
        rows=rows,
        trials=all_rows,
        complete=True,
        total_seconds=time.perf_counter() - start,
    )
    if out:
        save_json(out, result)
    return result


def summarize_random_trials(trials, setup, k):
    rows = []
    for j in range(k):
        fs = [r[j]["f"] for r in trials]
        times = [r[j]["seconds"] for r in trials]
        rows.append(
            dict(
                k=j + 1,
                f=float(np.mean(fs)),
                std=float(np.std(fs, ddof=1)) if len(trials) > 1 else 0.0,
                seconds=setup + sum(times),
                mean_trial_seconds=float(np.mean(times)),
                peak_mib=rss(),
            )
        )
    return rows


def bellman_reference(P, s, direction):
    """Endpoint policy improvement, certified afterwards by Bellman residual."""
    start = time.perf_counter()
    solver = Solver()
    a = np.full(len(s), 0.5)
    for it in range(1000):
        z = solver.solve(P, a, s)
        p = P @ z
        new = np.where(direction * (s - p) >= 0, HIGH, LOW)
        if np.array_equal(new, a):
            break
        a = new
    else:
        raise ArithmeticError("Reference policy iteration did not stabilize")
    z = solver.solve(P, a, s)
    p = P @ z
    alternatives = np.stack([LOW * s + (1 - LOW) * p, s])
    H = np.max(alternatives, axis=0) if direction == 1 else np.min(alternatives, axis=0)
    residual = float(np.max(abs(z - H)))
    bound = len(s) * residual / LOW
    if bound > 1e-5:
        raise ArithmeticError(f"Insufficient reference accuracy: {bound}")
    return dict(
        f=float(z.sum()),
        alpha=a,
        bellman_residual=residual,
        objective_error_bound=bound,
        iterations=it + 1,
        seconds=time.perf_counter() - start,
    )


LOCAL_TOL = 1e-5  # Absolute error in the TOTAL equilibrium opinion, not its mean.
LOCAL_START_NAMES = ("upper", "initial", "x_midpoint", "x_random")


def local_starts(a):
    """Canonical paper start followed by three chosen supplementary starts.

    Invert the midpoint/random X vectors; do not substitute an alpha midpoint.
    The same stream and arrays are used for both profiles and both objectives.
    """
    vectors = (
        np.full(len(a), HIGH),
        a.copy(),
        np.full(len(a), 2 / (1 + 1 / LOW)),
        1 / rng(3000).uniform(1, 1 / LOW, len(a)),
    )
    return list(zip(LOCAL_START_NAMES, vectors))


def assess_local_policy(P, s, a, direction, equilibrium):
    """Use fresh original-opinion solves to assess the final policy."""
    n = len(s)
    roundoff = 8 * np.finfo(float).eps
    actual, actual_error, solve_residual = equilibrium(a, s, direct=True)
    opinions = 1 - s if direction == 1 else s
    transformed, _, _ = equilibrium(a, opinions, direct=True)
    p = P @ actual
    alternatives = np.stack([LOW * s + (1 - LOW) * p, HIGH * s + (1 - HIGH) * p])
    H = np.max(alternatives, axis=0) if direction == 1 else np.min(alternatives, axis=0)
    bellman_residual = float(np.max(abs(H - actual)))
    bound = n * (bellman_residual + roundoff) / LOW + n * actual_error
    transformed_total = (
        float(n - transformed.sum()) if direction == 1 else float(transformed.sum())
    )
    return dict(
        f=float(actual.sum()),
        endpoint_count=int(np.count_nonzero((a == LOW) | (a == HIGH))),
        equilibrium_residual=solve_residual,
        bellman_residual=bellman_residual,
        objective_error_bound=bound,
        complement_error=abs(float(actual.sum()) - transformed_total),
    )


def local_search(
    P, s, a0, ids, direction, max_moves=None, objective_tolerance=LOCAL_TOL
):
    """Section 6.1 single-coordinate local search with checked equilibrium solves.

    This is the basic equilibrium-based strategy, not the approximate/optimistic
    algorithms in Section 6.3. Extend the paper's u<1 rule to u=1 using s-Pz:
    dz/dalpha_i = M^-1 e_i (s_i-(Pz)_i). No division by 1-alpha_i.
    Maximise by minimising complemented opinions. Never perturb the inputs.
    Numerical residual bounds are floating-point diagnostics, not interval
    proofs. An unresolved move/sign is not by itself evidence of optimality.
    """
    start = time.perf_counter()
    s = np.asarray(s, dtype=float)
    a = np.array(a0, dtype=float, copy=True)
    ids = np.asarray(ids)
    n = len(s)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if (
        n == 0
        or a.shape != s.shape
        or ids.shape != s.shape
        or P.shape != (n, n)
        or len(np.unique(ids)) != n
        or not np.isfinite(s).all()
        or not np.isfinite(a).all()
        or np.any((s < 0) | (s > 1))
        or np.any((a < LOW) | (a > HIGH))
    ):
        raise ValueError("Invalid opinions, resistances, IDs or matrix dimensions")
    if objective_tolerance <= 0:
        raise ValueError("objective_tolerance must be positive")
    canonical = bool(np.all(a == HIGH))
    max_moves = (n if canonical else 10 * n) if max_moves is None else max_moves
    if max_moves < 0:
        raise ValueError("max_moves must be nonnegative")
    opinions = 1 - s if direction == 1 else s
    order = np.argsort(ids, kind="stable")
    solver = Solver()
    direct_solves = 0
    history = []
    roundoff = 8 * np.finfo(float).eps

    def equilibrium(alpha, values, direct=False):
        nonlocal direct_solves
        M = system(P, alpha)
        if direct:
            direct_solves += 1
            z = np.asarray(spsolve(M, alpha * values))
        else:
            z = solver.solve(P, alpha, values)
        residual = float(np.max(abs(M @ z - alpha * values)))
        if not np.isfinite(z).all() or not np.isfinite(residual) or residual > 1e-10:
            raise ArithmeticError("Local-search equilibrium solve failed")
        error = (residual + roundoff) / float(np.min(alpha))
        return z, error, residual

    def original_total(z):
        return float(n - z.sum()) if direction == 1 else float(z.sum())

    z, error, _ = equilibrium(a, opinions)
    initial = original_total(z)
    stop_reason = "move_limit"
    rejected = 0
    refinement_passes = 0
    while len(history) < max_moves:
        accepted = False
        # Retry unresolved scans with direct solves before stopping. Each scan
        # restarts in original-ID order, accepting one reliably improving move.
        for refined in (False, True):
            if refined:
                z, error, _ = equilibrium(a, opinions, direct=True)
                refinement_passes += 1
            difference = opinions - P @ z
            for v in order:
                if difference[v] > error:
                    endpoint = LOW
                elif difference[v] < -error:
                    endpoint = HIGH
                else:
                    continue  # Exact/uncertain tie: keep the current resistance.
                if a[v] == endpoint:
                    continue
                trial = a.copy()
                trial[v] = endpoint
                new_z, new_error, new_residual = equilibrium(
                    trial, opinions, direct=refined
                )
                improvement = float(z.sum() - new_z.sum())
                uncertainty = n * (error + new_error) + roundoff * n
                if improvement <= uncertainty:
                    rejected += 1
                    continue
                if canonical and (a[v] != HIGH or endpoint != LOW):
                    raise ArithmeticError(
                        "Canonical local search attempted a repeated/reverse flip"
                    )
                old = float(a[v])
                before = original_total(z)
                a = trial
                z = new_z
                error = new_error
                history.append(
                    dict(
                        move=len(history) + 1,
                        index=int(v),
                        node=int(ids[v]),
                        alpha_before=old,
                        alpha_after=endpoint,
                        f_before=before,
                        f=original_total(z),
                        gain=improvement,
                        gain_uncertainty=uncertainty,
                        equilibrium_residual=new_residual,
                        seconds=time.perf_counter() - start,
                    )
                )
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            stop_reason = "no_resolved_improving_move"
            break

    diagnostics = assess_local_policy(P, s, a, direction, equilibrium)
    certified = bool(
        diagnostics["objective_error_bound"] <= objective_tolerance
        and diagnostics["complement_error"] <= objective_tolerance
    )
    status = (
        "certified"
        if certified
        else ("move_limit" if stop_reason == "move_limit" else "numerically_unresolved")
    )
    return dict(
        initial=initial,
        alpha=a,
        initial_alpha=np.array(a0, copy=True),
        direction="max" if direction == 1 else "min",
        canonical=canonical,
        moves=len(history),
        max_moves=max_moves,
        status=status,
        stop_reason=stop_reason,
        history=history,
        objective_tolerance=objective_tolerance,
        **diagnostics,
        rejected_trials=rejected,
        refinement_passes=refinement_passes,
        equilibrium_solves=solver.calls + direct_solves,
        direct_solves=direct_solves,
        solver_recoveries=solver.recoveries,
        seconds=time.perf_counter() - start,
        peak_mib=rss(),
    )


def unbudgeted(P, a, s, ids, direction, out=None, max_moves=None):
    reference = bellman_reference(P, s, direction)
    runs = []
    result = dict(
        method="local",
        schema_version=1,
        reference=reference,
        runs=runs,
        objective_tolerance=LOCAL_TOL,
        complete=False,
    )
    if out:
        save_json(out, result)
    for number, (name, a0) in enumerate(local_starts(a)):
        if out:
            print(
                f"Local search: starting {name} ({number+1}/{len(LOCAL_START_NAMES)})",
                flush=True,
            )
        run = local_search(P, s, a0, ids, direction, max_moves=max_moves)
        run.update(
            start=number,
            start_name=name,
            reference_gap=direction * (reference["f"] - run["f"]),
        )
        runs.append(run)
        if out:
            save_json(out, result)
        if out:
            print(
                f"Local search: {name}: {run['status']}, moves={run['moves']}, f={run['f']:.10f}",
                flush=True,
            )
    result["start_spread"] = float(np.ptp([run["f"] for run in runs]))
    result["optimum"] = dict(
        f=runs[0]["f"],
        start_name="upper",
        objective_error_bound=runs[0]["objective_error_bound"],
    )
    result["complete"] = bool(
        all(
            run["status"] == "certified" and abs(run["reference_gap"]) <= LOCAL_TOL
            for run in runs
        )
        and result["start_spread"] <= LOCAL_TOL
    )
    if out:
        save_json(out, result)
    return result


def optimized_set(base, S, direction=1):
    choices = [
        (base.evaluate(S, values), values)
        for values in product((LOW, HIGH), repeat=len(S))
    ]
    return max(choices, key=lambda x: direction * x[0])


def paper_example():
    ids, _, P = graph_matrix(nx.complete_graph([1, 2, 3]))
    a = np.full(3, 0.1)
    s = np.array([1.0, 0.0, 0.0])
    base = FixedBase(State(P, a, s))
    sets = [[], [0], [2], [0, 1], [0, 2], [0, 1, 2]]
    fs = [optimized_set(base, S)[0] for S in sets]
    expected = [1, 29 / 11, 2999 / 2009, 86971 / 31009, 86971 / 31009, 2999 / 1001]
    assert np.max(abs(np.array(fs) - expected)) < 1e-10
    return dict(
        sets=[[int(ids[v]) for v in S] for S in sets],
        f=fs,
        nonsubmodular=dict(small=fs[4] - fs[1], large=fs[5] - fs[3]),
        nonsupermodular=dict(small=fs[2] - fs[0], large=fs[4] - fs[1]),
    )


def witnesses(P, a, s, ids, limit=10000):
    state = State(P, a, s)
    base = FixedBase(state)
    generator = rng(4000)
    found = {}
    cache = {}

    def value(S):
        key = tuple(sorted(S))
        if key not in cache:
            cache[key] = optimized_set(base, list(key))
        return cache[key]

    for attempt in range(limit):
        # Both empty-to-singleton and singleton-to-pair comparisons.
        sample = generator.choice(len(a), 3, replace=False).tolist()
        A = [] if attempt % 2 == 0 else [sample[0]]
        B = A + [sample[1]]
        v = sample[2]
        sets = [sorted(A), sorted(A + [v]), sorted(B), sorted(B + [v])]
        vals = [value(S)[0] for S in sets]
        small = vals[1] - vals[0]
        large = vals[3] - vals[2]
        gap = large - small
        kind = (
            "nonsubmodular"
            if gap > 1e-6
            else "nonsupermodular" if gap < -1e-6 else None
        )
        if kind and kind not in found:
            verified = []
            assignments = []
            error_bound = 0.0
            for S in sets:
                f, es = value(S)
                trial = a.copy()
                trial[S] = es
                z_fresh = Solver().solve(P, trial, s)
                actual = float(z_fresh.sum())
                residual = float(np.max(abs(system(P, trial) @ z_fresh - trial * s)))
                error_bound += len(a) * (residual + 10 * np.finfo(float).eps) / LOW
                if abs(actual - f) > 1e-7:
                    raise ArithmeticError("Witness solve mismatch")
                verified.append(actual)
                assignments.append(list(es))
            verified_gap = (verified[3] - verified[2]) - (verified[1] - verified[0])
            if abs(verified_gap) <= 10 * error_bound or (verified_gap > 0) != (
                kind == "nonsubmodular"
            ):
                continue
            found[kind] = dict(
                numerical_error_bound=error_bound,
                A=ids[A],
                B=ids[B],
                v=int(ids[v]),
                sets=[ids[S] for S in sets],
                assignments=assignments,
                f=verified,
                marginal_small=verified[1] - verified[0],
                marginal_large=verified[3] - verified[2],
                attempt=attempt + 1,
            )
        if len(found) == 2:
            break
    return dict(witnesses=found, attempts=attempt + 1, complete=len(found) == 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/facebook_combined.txt"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--method",
        choices=["fast", "I", "II", "random", "local", "witness", "example"],
        default="fast",
    )
    parser.add_argument("--profile", choices=["U", "P"], default="U")
    parser.add_argument("--direction", choices=["max", "min"], default="max")
    parser.add_argument("--pool", choices=["all", "shortlist"], default="all")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Local search only: cap per start (default n for upper, 10*n otherwise)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.method == "local" and args.pool != "all":
        parser.error("Local search is unbudgeted and requires --pool all")
    if args.max_moves is not None and (args.method != "local" or args.max_moves < 0):
        parser.error("--max-moves requires --method local and a nonnegative value")
    if args.method == "example":
        save_json(args.output / "paper_example.json", paper_example())
        return
    ids, degree, P, a, profiles, meta = load_inputs(args.dataset)
    s = profiles[args.profile]
    direction = 1 if args.direction == "max" else -1
    pool = np.arange(len(a)) if args.pool == "all" else shortlist(ids, degree)
    filename = (
        args.output / f"{args.method}_{args.profile}_{args.direction}_{args.pool}.json"
    )
    if args.method == "fast":
        fast(P, a, s, ids, pool, args.k, direction, filename)
    elif args.method in ("I", "II"):
        baseline(P, a, s, ids, degree, args.method, args.k, direction, pool, filename)
    elif args.method == "random":
        random_trials(
            P,
            a,
            s,
            ids,
            args.k,
            direction,
            pool,
            int(args.pool == "shortlist"),
            out=filename,
        )
    elif args.method == "local":
        result = unbudgeted(P, a, s, ids, direction, filename, args.max_moves)
        if not result["complete"]:
            raise ArithmeticError(f"Local search not certified; inspect {filename}")
    elif args.method == "witness":
        save_json(filename, witnesses(P, a, s, ids))


if __name__ == "__main__":
    main()
