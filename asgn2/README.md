# Assignment 2 — 23CS10012

Task 2 part 2 now implements the corrected paper's non-convex local search in
place of projected gradient descent, following the TA's 11 November clarification.
**This source revision has not been run or verified.** The existing results,
PDF and ZIP belong to the previous implementation and are not a submission of
this revision. Execution, verification and artifact generation are left to you.

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
python Assignment2_23CS10012_analysis_task4.py run --output results-local
python Assignment2_23CS10012_analysis_task4.py report --output results-local --report Assignment2_23CS10012_report_local.pdf
python Assignment2_23CS10012_analysis_task4.py archive --output results-local --report Assignment2_23CS10012_report_local.pdf --archive Assignment2_23CS10012_local.zip
```

Use a **new** results-local directory for this revision. Do not copy earlier
worker JSON files into it. Source/data hashes change when numerical scripts or
the report generator change. The driver intentionally refuses results from a
different source fingerprint. The guide describes the deferred reuse weaknesses.

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
  three original assignment starts converted from X to resistance coordinates.
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

RUN_STATUS.md distinguishes this unexecuted revision from the previous run.
IMPLEMENTATION_PLAN.md and REVIEW.md retain historical planning material.
