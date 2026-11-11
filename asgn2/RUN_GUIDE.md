# Run and verify the rewritten Assignment 2

This guide describes execution and acceptance criteria for the refactored scripts.
The numerical checks, workflow regressions and report-calculation comparisons have
been exercised. The full Facebook benchmark has not been rerun after refactoring;
existing results retain their original provenance and timing measurements.

The old `results/`, report PDF and ZIP remain historical artifacts. Do not submit
them with the new scripts. The new implementation replaces Task 2 part 2's PGD;
Task 2's fast greedy, baselines, Random and witnesses retain their algorithms.
Graph loading, seeded parameter generation and software metadata are now separate.
The report reads the recorded input arrays and experiment metadata.

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

Keep the single-thread environment settings for every execution and report step.
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
python -m unittest discover -s tests -v
```

Stop on an exception or a nonzero exit code. Expected successful output includes
`All independent small-graph checks passed`. Open
`checks-local/checks/validation.json`: `passed` should be true. Do not trust a
file left from an earlier failed invocation without checking the command's exit.

The `unittest` command is for a source checkout. The numerical `checks` command
also works in the five-script submission archive.

The numerical suite checks:

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

Workflow regressions cover interrupted Random checkpoints, truncated completed
runs, missing trials, invalid local diagnostics, saved input ordering, source
completion markers and the meaning of maximisation/minimisation gain ratios.

For a code-level inspection, check the update rule in `local_search`: for the
minimizing opinions, d=s-Pz positive selects LOW, negative selects HIGH. Each
accepted move changes just one coordinate. Maximization uses 1-s, and the final
reported objective is freshly solved with the original s. The report documents
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
python Assignment2_23CS10012_analysis_task4.py run --output results-refactor
```

Choose an unused directory name if results-refactor already holds an unrelated or
partial manually assembled run. Keep smoke-local separate. The full driver
reruns the checks, Task 1, budgeted comparisons, all four local-search scenarios,
witnesses, Task 3 and the bonus. There are 16 local-search runs in total:
U/P x min/max x four starts.

Expected new result files include:

```text
results-refactor/local_U_max_all.json
results-refactor/local_U_min_all.json
results-refactor/local_P_max_all.json
results-refactor/local_P_min_all.json
results-refactor/inputs.npz
results-refactor/manifest.json
results-refactor/commands.json
results-refactor/checks/validation.json
results-refactor/complete.json
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

out = Path('results-refactor')
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

## 6. Generate and inspect the report and archive

Use the **same activated environment and thread settings** as the experiment:

```bash
python Assignment2_23CS10012_analysis_task4.py report --output results-refactor --report Assignment2_23CS10012_report_local.pdf
python Assignment2_23CS10012_analysis_task4.py archive --output results-refactor --report Assignment2_23CS10012_report_local.pdf --archive Assignment2_23CS10012_local.zip
```

The archive command itself runs checks after extraction, checks entry-point help,
and independently solves one Facebook equilibrium. Expected terminal messages
are `Wrote ...pdf` and `Archive extracted and validated: ...zip`.

Open the new PDF and inspect every page. Check for clipping, unreadable tables,
missing figures, and these content requirements:

- Task 2 part 2 explains the TA clarification, Section 6.1 method, u=1 adaptation,
  tie handling and numerical stopping criteria; it does not present PGD as the
  implemented method or claim convexity of the transformed objective.
- All four profile/objective cases show all four starts and their actual values,
  moves, statuses, error bounds and reference gaps.
- Plot reference lines use the canonical local result. Dependent fractions and
  comparisons use the new result directory.
- Task 3's restricted C optima are still distinguished from all-node optima.
- Software/seed/command information matches your execution manifest. The report
  uses the recorded experiment metadata and saved input arrays.

Inspect the ZIP contents: five Python scripts, requirements, README, RUN_GUIDE,
the **new** PDF, dataset and new results under results/. It should not contain
old pgd_* results, the old report, a virtual environment or a second stale ZIP.
If your submission requires the exact original PDF/ZIP names, use those names in
the final report/archive commands only after preserving the old artifacts and
checking the new ones. Do not upload the previous committed artifacts by mistake.

## Failure handling and deferred issues

| Symptom | Action |
| --- | --- |
| No module named resource | You are running native Windows Python; activate the Ubuntu environment. |
| Syntax/import/check failure | Stop; preserve the full traceback. Do not proceed to Facebook runs. |
| Local status move_limit | Inspect moves and objective history; diagnose with a higher cap in a separate directory. |
| Local status numerically_unresolved | Inspect rejected_trials, refinement_passes and residuals. Do not label it converged or relax tolerances to force success. |
| Large reference gap/spread | Compare all four runs, final independent solves and certificates; do not generate submission claims from that case. |
| Source/input hash mismatch | Use a fresh output directory and rerun. Do not edit hashes to bypass the guard. |
| Interrupted full run | Preserve logs; assess the worker files before resuming, as described below. |
| Missing aggregate rows / report crash | Inspect Random checkpoints and worker lengths; do not change complete flags by hand. |

The refactor resolves three previously deferred issues:

1. Worker reuse validates the requested budget rows. Ground-truth worker reuse
   also requires its companion files, and Random requires all 20 complete trials
   plus consistent aggregate objectives. Invalid checkpoints are rerun.
2. Random trial checkpoints remain `complete=false` until aggregate rows exist.
3. Reports obtain experiment metadata from `manifest.json` and inputs from
   `inputs.npz`. Rendering does not regenerate the experimental draw.

The remaining manual check is that the report is a valid, readable PDF. Archive
extraction and numerical checks do not establish page layout or PDF validity;
open the generated PDF and its copy inside the ZIP yourself.

Do not expect precisely the earlier runtimes or endpoint vectors. The intended
acceptance is a successful execution plus independently checked objectives and
documented numerical tolerances. Keep logs/tracebacks for any failed criterion.
