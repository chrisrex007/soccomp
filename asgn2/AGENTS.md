# Assignment 2 agent guidance

## Sources and scope

Read `Assignment2_OM_ego-Facebook.pdf` and `TA_GUIDE.md` first. The TA replaces
Task 2 part 2's PGD/convexity instruction with the corrected paper's non-convex
method. The other assignment requirements remain. Current code determines actual
behavior; `RUN_GUIDE.md` describes the manual execution workflow and `ISSUES.md`
tracks implementation status. These Markdown files do not override the PDF/TA.

Work only in `asgn2/`, on `main`. Keep the five prescribed script names and
sibling imports. Do not merge another branch wholesale. Keep the data, provenance,
assignment PDF, TA clarification, requirements, and environment intact.

The user explicitly requires **no execution by the agent**: do not run Python,
compilation, CLI help, tests, experiments, installs, or server syncs. Read files,
edit requested source/docs, and inspect diffs only. The user will run the commands
in `RUN_GUIDE.md`. Do not claim runtime correctness from source inspection.

Implement only the assignment/TA requirements. No extra regression suite,
independent optimizer, inverse/update verification passes, or numerical
certification framework. Ordinary solver failure handling, required budget/count
conditions, and the numerical algorithms' stopping rules remain necessary.

The report is manual. Do not add report-text generation/refresh, PDF generation,
or archive packaging. Required numerical CSVs and plots remain part of Task 4.
A static report checklist is deferred; do not create it as part of these fixes.
File deletion/commit housekeeping is not part of the issue review.

## Model and required work

- Use the undirected simple ego-Facebook graph: 4,039 nodes, 88,234 edges, one
  component. Record preprocessing and use original SNAP IDs for target sets.
- Solve `(I-(I-A)P)z=A s`, maximizing/minimizing `sum(z)`, not its mean.
- Resistance bounds are 0.001 and 1. Original resistances are uniform; opinions
  are U = Uniform[0,1] and P = Beta(0.5,0.5). Untargeted resistances stay original.
- Budget is 30; objective curves cover 1,2,3,4,5,10,20,30.
- Task 1: three two-node checks; Facebook iteration/solve agreement within
  `1e-8`; 21-point sweeps on three graphs of at most eight nodes; fresh-solve
  greedy on the top-200 shortlist through k=10 for both profiles, with counts/time.
- Task 2: Sherman–Morrison greedy for U-max/P-max/U-min through k=30, matching
  Task 1 on its shortlist; corrected unbudgeted max/min for U/P from three starts;
  independent exhaustive endpoint checks on two graphs of at most 15 nodes;
  Baselines I/II, Random with 20 trials, Example 13 and Facebook marginal evidence.
- Task 3: union of top-15 degree and top-15 single-node gain nodes, exhaustive
  size-2/3 endpoint assignments, original-ID optimum, counts, overlap and timing.
- Task 4: required plots and table at k=5,10,30, restricted/all-node gain ratios
  and Jaccards, C-restricted greedy, target-opinion plots and increasing marginals.
  The bonus uses improving endpoint flips/top-200-shortlist swaps.
- Permitted dependencies are NumPy, SciPy, NetworkX, pandas, matplotlib and stdlib;
  CVXPY is allowed only for Task 2 small-graph checks. Keep current dependency pins.

## Implementation map

All task scripts have prefix `Assignment2_23CS10012_`:

| Suffix | Responsibility |
| --- | --- |
| `greedy_task1.py` | Shared model/RNG/input/solver/JSON primitives, naive greedy, Task 1 small checks |
| `paper_task2.py` | Rank-one/Woodbury state, fast greedy, baselines, Random, local search, example/witnesses |
| `groundtruth_task3.py` | C construction and exhaustive comparisons |
| `bonus_task4.py` | Required flip/swap improvement, with a completed 30-target input prerequisite |
| `analysis_task4.py` | `checks`, `prepare`, fresh `run`, numerical `analyze` |

Preserve seed 2026 and streams 0/1/2 for resistance/U/P, 3000 for random-X,
4000 for witnesses, and `1000+100*pool_code+trial` for Random. Sorted original
IDs define matrix order. Reuse existing numerical primitives rather than adding
helper modules or alternate algorithms.

Local search uses `s-Pz` at alpha=1, with maximization through complemented
opinions and a final original-opinion solve. Starts are exactly `upper`, `initial`,
`x_random`; U/P × max/min gives 12 runs. Improvement threshold is `TIE=1e-9`;
three-start and small-graph objective comparison tolerance is `LOCAL_TOL=1e-5`.
These tolerances are implementation choices, not prescribed by the assignment.
Convergence labels are numerical, not certificates. No move-cap option or
Bellman-reference optimizer is supported. Tied supplementary policies can retain
interior coordinates. Local output schema version is 2.

Full runs require an absent/empty directory and always execute all workers.
There is no resume, cache migration or worker reuse. Only successful completion
of every requested worker writes the final marker. Retain manifest, source/data
fingerprint, actual command and software/version metadata for reproducibility.

Random's full-run peak memory is top-level `peak_mib`. Its k=5 and k=10 table
measurements use separate processes under `measurements/k5/` and `measurements/k10/`;
the normal k=30 run supplies k=30. Trial checkpoints are always incomplete until
aggregate rows are saved. Main CSV: 36 rows; local CSV: 12; groundtruth CSV: 32.
Do not restore obsolete certificate fields or old-schema compatibility branches.

Preserve full versus C-restricted comparison scope. Baselines set HIGH and use
complemented rankings for minimization. Random selects endpoints sequentially,
not by joint optimization. Target plots use pre-intervention neighbor equilibrium
opinions. Time/peak RSS include setup; document these choices accurately.

## User-run workflow

See `RUN_GUIDE.md` for environment setup, commands, expected files and failures.
The user runs `checks`, `prepare`, `run --output results-new`, then
`analyze --output results-new`, with single-thread BLAS settings and assertions
on. Driver and individual CLI defaults differ, so specify output paths explicitly.

For a separately requested server sync, the user's preference is an exact mirror:

```bash
rsync -a --delete ~/acads/soccomp/asgn2/ 23CS10012@10.5.18.116:~/soccomp/asgn2/
```

Do not execute sync as part of implementation or review.
