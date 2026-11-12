# Manual report workflow — 12 November 2026

Assignment2_23CS10012_report_notes.md is a fixed reference containing the report
explanations and saved numerical facts. It is not refreshed by Python. The user
will rerun the task scripts, compare their outputs with the notes and write the
report manually.

The driver now has only `checks`, `prepare`, `run` and `analyze`. Notes formatting,
report generation and archive packaging were removed. `analyze` retains the
comparison calculations, three CSV tables, analysis.json and twelve PNG plots.
The duplicate REPORT_GUIDE.md template was removed; its explanations remain in
the complete report notes.

## Saved execution evidence

Before the report-workflow edits, the dataset and all five Python files hashed
to `e9e8c52c2dc6c346229b50afe8d69ef0f9e9462bbece83862f3970da9874ea58`,
matching the saved manifest and completion marker in results-local/. That
folder contains all 16 corrected-local-search runs. The older results/ folder
contains historical PGD experiments.

The saved canonical objectives are U-max 3904.478179, U-min 132.539702,
P-max 3994.183406 and P-min 47.759809. All 16 starts have status certified and
agree within 1e-5 for each scenario. These numbers describe that saved execution.
The four numerical task scripts and dataset were unchanged by the workflow edits.
The full-source fingerprint changes when analysis_task4.py changes; use a fresh
result directory for your new full run.

## Verification

During the first report-workflow change, the independent small-graph checks
passed and all 96 budget comparison rows, 16 local-start rows, 32 restricted
reference comparisons and 12 plots were checked against the saved records.

For the subsequent removal of notes-generation and packaging code, syntax and
command-line help were checked, and analysis was run against an isolated copy
of the saved results. The three CSV tables and numerical analysis matched the
previous exports, all twelve plots were produced, and a Markdown sentinel was
left untouched. No full Facebook experiment or PDF generation was run.

Verification used host Python 3.12.3 with the existing pinned packages in
.venv/lib/python3.12/site-packages. The saved execution manifest retains its
original Python 3.12.14 environment.

## Existing execution limitations

Generic worker reuse can accept short runs, and Random can mark a checkpoint
complete before aggregate rows exist. Keep smoke/custom-budget outputs separate
and use a fresh directory for a new full execution. Analysis checks 30-round
all-node aggregates and all twenty Random trial lengths, and requires the 16
saved local runs to meet their convergence criteria.
