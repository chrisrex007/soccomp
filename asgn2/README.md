# Assignment 2 — 23CS10012

Python 3.12.14. Run from this directory; all five task scripts must remain together.
The scripts implement the equilibrium, optimization algorithms and baselines using
NumPy/SciPy numerical primitives. No opinion-dynamics package is used.

## Setup and reproduction

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/soccomp-matplotlib XDG_CACHE_HOME=/tmp/soccomp-cache
python Assignment2_23CS10012_analysis_task4.py checks
python Assignment2_23CS10012_analysis_task4.py prepare
python Assignment2_23CS10012_analysis_task4.py run
python Assignment2_23CS10012_analysis_task4.py report
python Assignment2_23CS10012_analysis_task4.py archive
```

`prepare` downloads and validates the official SNAP dataset, requiring HTTPS access
to `snap.stanford.edu`. Alternatively place the original decompressed
`facebook_combined.txt` in `data/`. Graph counts must be 4,039 nodes, 88,234 edges,
and one connected component. All original node labels are retained.

The default `results/` directory contains raw numerical records, input arrays,
checksums, command history and generated figures/tables. Source/data hashes guard
against resuming with different experimental inputs. To rerun changed source or
parameters, use `--output results-new`. Completed workers are reused only within
that matching experiment. Interrupted individual workers restart from round one;
partial results never count as completed experiments.

`report` and `archive` require the completed experiment marker and matching source
and dataset hashes. The archive validator extracts into a temporary directory,
checks all entry points, runs the small-graph suite and independently solves one
Facebook equilibrium. A report is not generated from synthetic placeholder data.

Each task script has `--help`. Example standalone commands:

```bash
python Assignment2_23CS10012_greedy_task1.py --profile U
python Assignment2_23CS10012_paper_task2.py --method fast --profile P
python Assignment2_23CS10012_paper_task2.py --method pgd --profile U --direction min
python Assignment2_23CS10012_groundtruth_task3.py --profile U --restricted-comparisons
python Assignment2_23CS10012_bonus_task4.py --profile U --direction max
```

## Conventions

- Resistances: Uniform[0.001,1], shared across U/P profiles. Opinions: U uniform,
  P Beta(0.5,0.5). Independent SeedSequence streams rooted at 2026.
- Greedy retains earlier endpoint assignments. Task 1 solves every candidate from
  scratch on the full graph, restricted only in its 200-node candidate pool.
- Baselines I and II use the paper's rankings and assign selected resistance 1.
  Minimisation applies their rankings to complemented innate opinions.
- Random uses 20 independently seeded permutation prefixes. Each new target gets
  its better endpoint against the current prior assignments. This is sequential,
  not joint endpoint optimization.
- The paper's relevant example is numbered 4; the assignment calls it 13. Its
  printed supermodularity marginal also omits subtracting the initial objective.
- The transformed objective in X is not convex in general. The requested PGD runs
  remain part of the experiments. Independent endpoint-policy/Bellman checks
  certify numerical reference values; a polynomial-size LP in value coordinates
  provides a separate theoretical argument for unbudgeted solvability.
- Exhaustive Task 3 optima are restricted to C. Comparisons with all-node methods
  may exceed a gain ratio of one. Numerical ties are resolved by original IDs.
- Runtime includes initialization; main measured algorithms use fresh Python
  processes. RSS is total process peak in MiB, not only Python allocations.

## Completed results

Full Facebook experiments are complete. Both ten-round shortlist runs match fast
greedy; all twelve PGD starts reach their independently certified reference values.
The exact pools have 26 nodes (U) and 24 nodes (P). Greedy matches their optima at
k=2 and k=3. Both distinct modularity violations are witnessed for both profiles.
Greedy beats Baselines I and II at every required budget in all three scenarios.
The bonus neighborhood scan finds no improvement in all 24 tested cases.

The PDF report, submission ZIP and raw results are included. See `RUN_STATUS.md`
for the validation record. `IMPLEMENTATION_PLAN.md` and `REVIEW.md` preserve the
planning/review history; their planning-only status statements are historical.

`results/execution_provenance.json` records a presentation-only revision after
numerical execution: the report text, bibliography and plotting layout were
improved while all four numerical scripts and the analysis module outside its
report function were verified unchanged. Both execution and delivery hashes are
preserved. The corrected paper is Abebe et al. (2020), arXiv:2011.02583v1.
