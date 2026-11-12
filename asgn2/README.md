# Assignment 2 — 23CS10012

Use [Assignment2_23CS10012_report_notes.md](Assignment2_23CS10012_report_notes.md)
as a fixed reference while writing your report. It contains the equations,
explanations, saved results, comparisons, timings, witnesses and plot guidance.
Rerun the task scripts and check the notes against your new outputs yourself.
No script generates or refreshes the Markdown notes or report PDF.

`Assignment2_23CS10012_analysis_task4.py` contains validation, experiment
execution, comparison calculations, CSV exports and PNG plots. Its commands are
`checks`, `prepare`, `run` and `analyze`. Notes formatting, report generation and
archive packaging have been removed.

## Run the assignment

Use Linux/WSL and keep all five task scripts together. Activate your environment
and install requirements.txt if necessary. From this directory:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/soccomp-matplotlib XDG_CACHE_HOME=/tmp/soccomp-cache
python Assignment2_23CS10012_analysis_task4.py checks --output checks-new
python Assignment2_23CS10012_analysis_task4.py prepare
python Assignment2_23CS10012_analysis_task4.py run --output results-new
python Assignment2_23CS10012_analysis_task4.py analyze --output results-new
```

Use a fresh directory for a new numerical execution. The run command enforces
the source/data fingerprint; do not copy worker files from another revision.
`prepare` validates data/facebook_combined.txt or downloads the official SNAP
file if absent. Expected counts: 4,039 nodes, 88,234 edges and one component.

`analyze` reads completed results and writes:

- `comparison_table.csv`: all eight budgets for four methods in U-max, P-max
  and U-min, with signed improvement and Random standard deviation.
- `groundtruth_comparisons.csv`: gain ratios and Jaccards, with all-node and
  C-restricted pools separated.
- `local_search_comparisons.csv`: all 16 unbudgeted starts.
- `analysis.json`: source provenance, signed baseline gaps and marginal counts.
- `figures/`: objective, runtime, endpoint, target and bonus plots.

To plot the existing saved run, use `analyze --output results-local`. That
folder contains the corrected local-search experiments; the older `results/`
contains PGD experiments. The notes describe the saved local-search run, so
compare its snapshot with your new execution before using numbers in your PDF.

See [RUN_GUIDE.md](RUN_GUIDE.md) for staged commands and acceptance checks, and
[RUN_STATUS.md](RUN_STATUS.md) for saved execution evidence and verification.
