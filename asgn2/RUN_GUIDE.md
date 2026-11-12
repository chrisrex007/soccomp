# Run experiments and prepare material for your own report

The report PDF is written manually. The Markdown notes are a fixed reference;
check them against your own reruns. Use `analyze` for CSVs and plots only. The saved
`results-local/` contains completed corrected-local-search experiments; the
older `results/` contains historical PGD experiments. See RUN_STATUS.md for
provenance and the scope of verification in the current change.

To compute tables and plots from the existing saved results, after activating
your environment:

```bash
python Assignment2_23CS10012_analysis_task4.py analyze --output results-local
```

The sections below describe an optional **new** numerical execution. Use a new
result directory, such as `results-new`, for this source revision.

## 1. Open Ubuntu and prepare Python

From PowerShell:

```powershell
wsl.exe -d Ubuntu --cd /mnt/c/Users/krish/soccomp/asgn2
```

All subsequent commands in this guide are **Bash commands inside Ubuntu**, run
from `asgn2/`. If you use a Linux-native clone, change to its `asgn2/` instead.
Do not reuse a Windows-created virtual environment.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/soccomp-matplotlib XDG_CACHE_HOME=/tmp/soccomp-cache
```

The previous run used Python 3.12.14 and the package versions already pinned in
requirements.txt. Prefer Python 3.12; if necessary create the venv with
`python3.12 -m venv .venv`. If Ubuntu reports that venv/ensurepip is missing,
install the matching Ubuntu Python venv package before retrying. If pip cannot
resolve the pinned versions, stop and record the error rather than silently
changing dependencies or substituting old experiment files.

Keep the single-thread environment settings for every execution and analysis step.
Record any deliberate change of Python/package versions. The new run's manifest
records the actual software environment; old timing numbers are not predictions
for your laptop. Dense inverses used by the other assignment tasks require
substantial memory (one 4039-by-4039 float64 matrix alone is about 125 MiB).

## 2. Check syntax and entry points, then small graphs

Do not run Python with `-O`: the existing suite uses assertions.

```bash
python -m py_compile Assignment2_23CS10012_*.py
for script in Assignment2_23CS10012_*.py; do
    python "$script" --help || break
done
python Assignment2_23CS10012_analysis_task4.py checks --output checks-local
```

Stop on an exception or a nonzero exit code. Expected successful output includes
`All independent small-graph checks passed`. Open
`checks-local/checks/validation.json`: `passed` should be true. Do not trust a
file left from an earlier failed invocation without checking the command's exit.

The suite now checks:

- Path and complete graphs: all endpoint assignments enumerated with independent
  dense linear solves, for minimization and maximization. Local search runs from
  all four starts and must match the exhaustive value within absolute `1e-5`.
- Every accepted move replayed with an independent solve, preserving feasibility
  and the expected objective direction; logged gain exceeds logged uncertainty.
- Upper-bound start: at most n moves, all final coordinates are endpoints, and
  no node is flipped twice or from low back to high.
- Two connected nodes with s=(1,0): from all-upper resistance, the minimum is
  **0.001** and maximum **1.999**. This catches the false stationary condition
  caused by testing s-z at alpha=1. Both runs should take one accepted move.
- Constant opinions (0, 0.5 and 1): zero moves and unchanged interior resistance
  vectors are valid. Do not require endpoint assignments in these tie cases.
- Complement identity, original-ID ordering after a matrix permutation, and a
  zero-move cap returning an uncertified result on a non-optimal policy.
- The existing equilibrium, greedy, Woodbury, Random, exhaustive and bonus checks.

For a code-level inspection, check the update rule in `local_search`: for the
minimizing opinions, d=s-Pz positive selects LOW, negative selects HIGH. Each
accepted move changes just one coordinate. Maximization uses 1-s, and the final
reported objective is freshly solved with the original s. The report notes explain
why the paper's u<1 rule needs this adaptation for the assignment's u=1.

## 3. Validate the dataset and try one Task 2 case

```bash
python Assignment2_23CS10012_analysis_task4.py prepare
python Assignment2_23CS10012_paper_task2.py --method local --profile U --direction min --output smoke-local
```

The supplied dataset is validated rather than downloaded again if it exists.
Expected graph: 4,039 nodes, 88,234 edges, one connected component. If absent,
prepare needs HTTPS access to snap.stanford.edu.

The local command prints the start name before each run and its status, move
count and objective afterwards. A start can take time because each candidate
move needs an equilibrium solve; no runtime estimate has been measured for this
revision. The JSON checkpoint is updated **after a start finishes**, not after
every individual move, and remains incomplete until all four starts pass.

Inspect `smoke-local/local_U_min_all.json`. The four starts, in order, are:

| Name | Initial resistance | Purpose |
| --- | --- | --- |
| upper | all 1 | Canonical Section 6.1 run |
| initial | sampled assignment resistance | First original starting point |
| x_midpoint | all 1/500.5 | Midpoint of the original X box, inverted |
| x_random | 1/X, X sampled uniformly from [1,1000], stream 3000 | Third original starting point |

The command intentionally runs all four, so the original three-start comparison
is retained in addition to the paper's canonical start. A midpoint of alpha=0.5005
would be a different experiment and is not used.

Successful output must meet **all** of these conditions:

| Field/check | Expected |
| --- | --- |
| method / complete | local / true |
| runs | exactly four, with the names above |
| each status | certified |
| each objective_error_bound | <= 1e-5, absolute total-opinion units |
| each absolute reference_gap | <= 1e-5 |
| start_spread | <= 1e-5 |
| each equilibrium_residual | <= 1e-10 |
| each complement_error | <= 1e-5 |
| alpha and initial_alpha | 4,039 finite entries in [0.001,1] |
| upper endpoint_count / moves | 4,039 / at most 4,039 |
| each history entry | gain > gain_uncertainty, positive gain |

The final reference line comes from `optimum.f`, which is the canonical local
run's value. `reference.f` is the independent simultaneous policy-improvement
cross-check. Equal objectives are expected within tolerance, not bit-for-bit;
resistance vectors need not be identical when there are ties. Supplementary
starts can retain interior coordinates. An uncertified run writes diagnostics
with `complete=false` and the standalone command exits with an error.

The objective is a sum, between 0 and 4039, not a mean. For the canonical start,
the initial objective is sum(s). Max histories should increase and min histories
should decrease. Both U and P are fixed seeded opinion draws, not repeated samples.

## 4. Run the complete assignment into a fresh directory

After the small checks and smoke run pass:

```bash
python Assignment2_23CS10012_analysis_task4.py run --output results-new
```

Choose an unused directory name; results-local already holds the saved run. Keep smoke-local separate. The full driver
reruns the checks, Task 1, budgeted comparisons, all four local-search scenarios,
witnesses, Task 3 and the bonus. There are 16 local-search runs in total:
U/P x min/max x four starts.

Expected new result files include:

```text
results-new/local_U_max_all.json
results-new/local_U_min_all.json
results-new/local_P_max_all.json
results-new/local_P_min_all.json
results-new/inputs.npz
results-new/manifest.json
results-new/commands.json
results-new/checks/validation.json
results-new/complete.json
```

The final complete marker should appear only after the driver succeeds. Inspect
all four local JSONs using the table above. `commands.json` should contain local
method invocations, and the new run should produce no pgd_* result files.

The default move limits are n for upper and 10*n for other starts. The latter is
a practical guard, not a proved bound. For diagnosis only, standalone local
commands accept `--max-moves N`. A larger cap helps only a move_limit failure;
it does not resolve numerical uncertainty. Diagnostic/custom-cap outputs should
use a separate directory and must not be copied into the full experiment.

## 5. Independently inspect the saved local solutions

Run this from asgn2 after the full run. It reads the saved opinions/resistances,
reconstructs the graph matrix, and solves the final policies directly; it does
not call local_search or the Bellman reference routine. Assertions should all
pass, and it prints one summary for each of the four scenarios.

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from Assignment2_23CS10012_greedy_task1 import load_inputs

out = Path('results-new')
ids, _, P, _, _, _ = load_inputs(Path('data/facebook_combined.txt'))
inputs = np.load(out / 'inputs.npz')
assert np.array_equal(ids, inputs['ids'])
n, low = len(ids), 0.001
names = ['upper', 'initial', 'x_midpoint', 'x_random']
for profile in ('U', 'P'):
    s = inputs[profile]
    for direction in ('max', 'min'):
        data = json.loads((out / f'local_{profile}_{direction}_all.json').read_text())
        assert data['complete'] and data['method'] == 'local'
        assert [r['start_name'] for r in data['runs']] == names
        assert data['start_spread'] <= 1e-5
        values = []
        for r in data['runs']:
            a = np.asarray(r['alpha'])
            assert a.shape == (n,) and np.isfinite(a).all()
            assert np.all((a >= low) & (a <= 1))
            assert r['status'] == 'certified'
            assert r['objective_error_bound'] <= 1e-5
            assert abs(r['reference_gap']) <= 1e-5
            M = sparse.eye(n, format='csr') - sparse.diags(1-a) @ P
            z = spsolve(M, a*s)
            residual = np.max(np.abs(M @ z - a*s))
            assert residual <= 1e-10
            assert abs(z.sum() - r['f']) <= 1e-5
            p = P @ z
            H = (np.maximum if direction == 'max' else np.minimum)(low*s+(1-low)*p, s)
            bound = n*np.max(np.abs(H-z))/low
            assert bound <= 1e-5
            complement = spsolve(M, a*(1-s))
            assert abs(z.sum()+complement.sum()-n) <= 1e-5
            trial = np.asarray(r['initial_alpha']).copy()
            assert trial.shape == (n,) and np.isfinite(trial).all()
            assert np.all((trial >= low) & (trial <= 1))
            sign = 1 if direction == 'max' else -1
            previous = r['initial']
            assert len(r['history']) == r['moves']
            for number, h in enumerate(r['history'], 1):
                assert h['move'] == number and ids[h['index']] == h['node']
                assert trial[h['index']] == h['alpha_before']
                assert h['alpha_after'] in (low, 1)
                trial[h['index']] = h['alpha_after']
                assert h['gain'] > h['gain_uncertainty'] >= 0
                assert sign*(h['f']-h['f_before']) > 0
                assert abs(h['f_before']-previous) <= 1e-5
                previous = h['f']
            assert np.array_equal(trial, a)
            assert abs(previous-r['f']) <= 1e-5
            if r['start_name'] == 'upper':
                assert np.all(np.asarray(r['initial_alpha']) == 1)
                assert r['moves'] <= n and r['endpoint_count'] == n
                assert len({h['node'] for h in r['history']}) == r['moves']
            values.append(float(z.sum()))
        assert np.ptp(values) <= 1e-5
        assert abs(data['optimum']['f']-values[0]) <= 1e-5
        print(profile, direction, 'verified; values =', values)
PY
```

For stronger full-graph history checking, independently solve each reconstructed
trial resistance vector inside that loop and compare with h['f']; that can be
expensive. The small-graph suite already does this for every move on its fixtures.

## 6. Export analysis, then write your PDF

```bash
python Assignment2_23CS10012_analysis_task4.py analyze --output results-new
```

Expected message: `Wrote analysis tables and plots to ...`. Use the PNGs under
`results-new/figures/` and the CSVs under `results-new/`.

`Assignment2_23CS10012_report_notes.md` is a fixed reference with explanations,
saved numerical tables, local starts, witnesses and plot guidance. Compare your
new outputs against it yourself and use your new values in the report when they
differ. The script does not read, generate or update that Markdown file.
There are no `--notes`, `--report` or `--archive` options.

Check these details before writing your final conclusions:

- Task 2 part 2 describes Section 6.1 local search, the u=1 adaptation, tie
  handling and numerical stopping criteria. It does not present PGD as the
  implemented method or claim the transformed objective is convex.
- Every local profile/objective case includes all four starts and their actual
  objectives, moves, statuses, bounds and reference gaps.
- Unbudgeted reference lines/fractions use the canonical local result.
  Task 3's C-restricted optima remain distinct from all-node optima.
- Software and timing descriptions use the **saved execution manifest**.
  The fixed notes retain the earlier run's provenance; analysis.json identifies
  the saved experiment hash and the source used for the latest analysis.

## Failure handling and existing execution limitations

| Symptom | Action |
| --- | --- |
| No module named resource | Use Linux/WSL Python. |
| Syntax/import/check failure | Preserve the traceback and fix it before a full run. |
| Local status move_limit | Inspect history; diagnose with a higher cap in a separate directory. |
| Local status numerically_unresolved | Inspect rejected trials, refinement passes and residuals. Do not relax tolerance just to label it converged. |
| Large reference gap/spread | Diagnose final solves before using that case in report claims. |
| Run source hash mismatch | Use a fresh directory and rerun. Do not edit hashes to bypass the guard. |
| Analysis source hash differs | analysis.json identifies both revisions; the tables describe the saved run. |
| Interrupted Random run | Check all 20 trial trajectories and aggregate rows before resuming. |

Existing execution limitations remain: generic worker reuse can accept short
runs, and Random can mark a checkpoint complete before aggregate rows exist.
Keep smoke/custom-budget results separate from a full run and prefer a fresh
run directory after interruption. Analysis now refuses incomplete 30-round
all-node trajectories and missing Random trial/aggregate rows, and requires
all 16 local runs to pass their saved convergence criteria.

Do not expect identical runtimes on a different machine. Keep logs and
tracebacks for any failed acceptance check.
