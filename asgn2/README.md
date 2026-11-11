# Assignment 2 — 23CS10012

Task 2 part 2 now implements the corrected paper's non-convex local search in
place of projected gradient descent, following the TA's 11 November clarification.
The scripts have been refactored into separate input, validation, analysis and
rendering functions. Numerical checks and workflow regression tests cover this
revision. Existing `results-local/` records a completed run before the refactor;
its results are preserved, but its source fingerprint does not match these scripts.
Run into a fresh directory before generating submission artifacts for this revision.

Read [RUN_GUIDE.md](RUN_GUIDE.md) for WSL setup, staged commands, expected outputs,
acceptance criteria, troubleshooting and the final submission checklist.

## Reproduction

Use Linux/WSL: the scripts use Linux resource memory accounting. The previous
execution used Python 3.12.14; requirements.txt preserves its pinned packages.
Keep all five task scripts together and run from this directory. Algorithms use
NumPy/SciPy primitives, not an opinion-dynamics package.

After creating and activating a Linux virtual environment and installing the
requirements, run:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/soccomp-matplotlib XDG_CACHE_HOME=/tmp/soccomp-cache
python Assignment2_23CS10012_analysis_task4.py checks --output checks-local
python Assignment2_23CS10012_analysis_task4.py prepare
python Assignment2_23CS10012_analysis_task4.py run --output results-refactor
python Assignment2_23CS10012_analysis_task4.py report --output results-refactor --report Assignment2_23CS10012_report_local.pdf
python Assignment2_23CS10012_analysis_task4.py archive --output results-refactor --report Assignment2_23CS10012_report_local.pdf --archive Assignment2_23CS10012_local.zip
```

Use a **new** results-refactor directory for this revision. Do not copy earlier
worker JSON files into it. Source/data hashes change when numerical scripts or
the report generator change. The driver intentionally refuses results from a
different or missing source fingerprint. Reused worker files must cover every
requested budget and, for Random, all 20 trials and their aggregate rows.

In a source checkout, also run the workflow regression tests:

```bash
python -m unittest discover -s tests -v
```

The five-script submission retains its self-contained `checks` command; the
additional source-checkout tests exercise interruption recovery and report math.

The prepare command validates the existing data/facebook_combined.txt, or
downloads the official SNAP file if absent. Expected counts: 4,039 nodes, 88,234
edges and one connected component. Original node labels are retained.

Standalone Task 2 example (separate directory from the full experiment):

```bash
python Assignment2_23CS10012_paper_task2.py --method local --profile U --direction min --output smoke-local
```

## Task 2 part 2

- The basic strategy follows Section 6.1 of Abebe et al. (2020),
  [arXiv:2011.02583v1](https://arxiv.org/html/2011.02583v1#S6).
  This is not the approximate/optimistic algorithm of Section 6.3.
- Minimise by solving equilibrium, comparing s - Pz, moving one improving
  coordinate to its preferred endpoint, and solving again. Scan by original ID.
  Using neighbour averages accommodates the assignment's upper bound of 1;
  the paper states its theory with upper bounds strictly below 1.
- Maximise by minimising complemented opinions, with final evaluation on the
  original opinions. No random perturbation is added to the assignment inputs.
- Four runs per profile/objective: the canonical all-upper-bound start plus the
  three supplementary starts chosen for this implementation, including two
  converted from X to resistance coordinates. The assignment does not prescribe
  particular starting vectors.
  Only the canonical run has the at-most-n irrevocable-flip guarantee.
- Keep exact or unresolved ties unchanged. Supplementary starts can retain
  interior coordinates at an optimum; endpoint counts are reported honestly.
- Independent final equilibrium solves, Bellman residual bounds, and the
  separate simultaneous policy-improvement reference check numerical accuracy.
  Floating-point diagnostics are not rigorous interval certificates.
- The PGD CLI option and implementation have been removed. The new report uses
  local-search results, not supplementary PGD results.

## Other assignment conventions

- Initial resistance: Uniform[0.001,1], shared across U/P. Opinions: U uniform,
  P Beta(0.5,0.5). Independent SeedSequence streams rooted at 2026.
- Task 1 evaluates candidates from scratch in its 200-node pool on the full
  graph. Greedy retains previous endpoint assignments.
- Baselines I and II use the original rankings and assign selected resistance 1.
  Minimisation applies their rankings to complemented innate opinions.
- Random uses 20 seeded permutation prefixes and chooses each new endpoint
  against earlier assignments; it does not jointly optimise the whole set.
- The original paper's relevant example is numbered 4, called 13 in the
  assignment; its printed supermodularity marginal omits the initial objective.
- Task 3 exhaustive optima are restricted to C. All-node comparison ratios can
  exceed one. Runtime includes initialization; RSS is total process peak in MiB.

Report calculations are independent of plotting. Reports use the saved
`inputs.npz` and experiment `manifest.json`, including the recorded software
versions, rather than generating new inputs or substituting the rendering environment.

RUN_STATUS.md distinguishes refactor verification from the previous full run.
IMPLEMENTATION_PLAN.md and REVIEW.md retain historical planning material.
