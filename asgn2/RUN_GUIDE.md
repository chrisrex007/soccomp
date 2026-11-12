# Run Assignment 2 manually

Use the assignment PDF and `TA_GUIDE.md` as the requirements. The unbudgeted
method is the corrected paper's Section 6.1 local search with equilibrium linear
solves and the assignment's upper-bound adaptation. The report is your manual
work. The scripts produce numerical JSON, CSV tables and PNG plots only.

The current implementation has **not been executed or tested during this change**.
The commands below are for you to run. No runtime or convergence claims are made.

## 1. Environment

Work from `asgn2/` on Linux/WSL or the Linux server. Use the appropriate Linux
virtual environment and the versions in `requirements.txt`. If needed, create
and install the environment yourself:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python --version
python -m pip check
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/soccomp-matplotlib XDG_CACHE_HOME=/tmp/soccomp-cache
```

Record your actual Python version. Do not reuse a Windows-created environment.
Keep the thread settings for every command. Dense inverses require substantial
memory: one 4039-by-4039 float64 matrix alone is about 125 MiB. No runtime estimate
has been established for this revision.

## 2. Required small-graph checks

Choose an unused output directory. Do not use Python `-O`: the assignment checks
use assertions.

```bash
python Assignment2_23CS10012_analysis_task4.py checks --output checks-new
```

Expected success message: `All required small-graph checks passed`. The output
is under `checks-new/checks/`:

- `task1_checks.json`: the two-node formula for three parameter choices and
  21-point endpoint sweeps on a five-node path, triangle, and six-node star.
- `paper_example.json`: the numerical reproduction of Example 13.
- `validation.json`: unbudgeted min/max compared with every endpoint assignment
  on a six-node path and five-node complete graph, from each of the three starts.
  The absolute objective comparison tolerance is `1e-5`.

Stop on a traceback or nonzero exit. A pre-existing `passed` file is not evidence
that a later command succeeded. There is no additional regression/certification
suite. The Facebook iteration comparison and ten-round shortlist comparison run
as part of the full experiment below.

## 3. Prepare the data

```bash
python Assignment2_23CS10012_analysis_task4.py prepare
```

This checks the existing `data/facebook_combined.txt` graph or downloads the
SNAP file if absent. Expected counts: 4,039 nodes, 88,234 undirected edges, one
component. Downloading requires network access. Original SNAP IDs are retained;
preprocessing counts and dataset/software metadata are recorded for a full run.

## 4. Full experiment — fresh directory only

```bash
python Assignment2_23CS10012_analysis_task4.py run --output results-new
```

`results-new` must not exist or must be empty. A file or nonempty directory is
rejected before any experiment output is written. **There is no resume or reuse.**
After interruption or failure, choose another empty directory and start again.
Do not copy worker results from a smoke run or another source revision into it.

The driver performs the required small checks, Facebook iteration/solve comparison
(within `1e-8`), Task 1 and its ten-round fast-greedy shortlist comparison, all
budgeted methods, unbudgeted local search, witnesses, Task 3, and the bonus.

The three local starts, in order, are:

| Name | Resistance vector |
| --- | --- |
| `upper` | All 1; canonical Section 6.1 start |
| `initial` | Sampled original Uniform[0.001, 1] vector |
| `x_random` | Reciprocals of Uniform[1, 1000], RNG stream 3000 |

U/P × max/min × three starts gives **12 local runs**. Each local result has
`schema_version=2`, `method=local`, three `runs`, `start_spread`, and `optimum`
using the upper-start result. Each run records `initial`, `f`, final `alpha`,
`initial_alpha`, `moves`, `endpoint_count`, `status`, `seconds`, and `peak_mib`.

`converged` means that the numerical search found no endpoint move improving
the transformed total by more than `1e-9`. It is not a residual certificate or
an exact-arithmetic proof. Completion requires the three final objectives to
agree within `1e-5` in total-opinion units. No Bellman optimizer, reference gap,
certificate, per-move validation history, or `--max-moves` option is present.
Ties may leave interior coordinates in supplementary starts.

Maximization minimizes complemented opinions; the reported final objective is
solved with the original opinions. The test uses `s-Pz` so that alpha=1 does not
falsely force stationarity. The original sampled opinions are not perturbed.

Random uses 20 seeded permutations with nested prefixes and sequential better-endpoint
choices. The normal k=30 runs supply the eight-budget objective curves. Six extra
fresh Random processes measure k=5 and k=10 for U-max, P-max, and U-min. Their
outputs are saved under:

```text
results-new/measurements/k5/random_U_max_all.json
results-new/measurements/k5/random_P_max_all.json
results-new/measurements/k5/random_U_min_all.json
results-new/measurements/k10/random_U_max_all.json
results-new/measurements/k10/random_P_max_all.json
results-new/measurements/k10/random_U_min_all.json
```

Random's final `peak_mib` and `total_seconds` describe the entire requested-budget
process: setup and all 20 trials. Aggregate/trial rows do not claim per-budget
peak memory. All trial-only checkpoints have `complete=false`; only the final
result containing aggregate `rows` and all trials is complete.

The full experiment records `manifest.json`, `inputs.npz`, `commands.json`,
`equilibrium.json`, task JSONs, and finally `complete.json`. There are 33 worker
commands, including the six extra measurements; local starts are internal to
four local-worker invocations. The completion marker is written only after all
workers succeed. Individual task CLIs default to `results/`; the driver defaults
to `results-local/`. Always pass an explicit output path to avoid confusion.

## 5. Numerical analysis

After a successful full run:

```bash
python Assignment2_23CS10012_analysis_task4.py analyze --output results-new
```

This reads completed outputs and produces:

- `comparison_table.csv`: **36 rows**, four methods × three scenarios × k=5,10,30.
- `groundtruth_comparisons.csv`: **32 rows**, two profiles × all/C pools × four
  methods × k=2,3, including gain ratios and Jaccards.
- `local_search_comparisons.csv`: **12 rows**, the three starts for four cases.
- `analysis.json`: increasing-marginal counts, signed greedy/baseline gaps, and
  analysis provenance.
- `figures/`: **12 PNGs**, including objective, runtime, target, bonus and endpoint
  sweep plots. Objective curves retain all eight required budgets.

Random table rows use the corresponding k=5/10/30 process's objective/statistics,
full elapsed time and process peak RSS. Other methods use their cumulative
prefix time and process high-water RSS at that prefix. Memory includes Python,
native arrays and setup, not just the marginal data structures.

Task 3's optimum is restricted to C. An all-node gain ratio can exceed one.
Jaccard is against the deterministically selected reference; Random averages
per-trial overlaps. Exact Task 3 objective ties retain the first configuration
in sorted ID/endpoint order. Greedy uses objective tolerance `1e-9`, then ID
and lower endpoint. Target plots use pre-intervention neighbor equilibrium
opinions, and increasing marginals use tolerance `1e-8`.

These output counts describe the intended code paths, not a verified execution.
Old output schemas are unsupported; generate a fresh run with these scripts.

## Failure handling

- A failed solver/factorization or nonfinite equilibrium stops execution.
- Disagreeing local starts leave `complete=false` and cause a nonzero exit.
  Inspect the actual objectives; do not relax tolerances merely to claim success.
- Failure to find both marginal witnesses also leaves an incomplete result.
- The bonus requires a completed 30-round greedy file with 30 distinct targets.
  Short, incomplete or duplicate-target input is rejected before bonus output.
- If the full run fails, preserve its traceback and choose a new directory for
  the next full run. Analysis requires the final completion marker and the
  matching manifest/dataset. It does not fill in missing experiments.
- Write the report yourself after inspecting the new outputs. The scripts do not
  generate or refresh report text, report PDFs, Markdown notes, or ZIP archives.
