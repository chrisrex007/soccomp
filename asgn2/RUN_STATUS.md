# Current revision — 11 November 2026: execution pending

Task 2 part 2 has been rewritten to use the corrected paper's basic local-search
strategy. The PGD implementation, driver calls and report sections are replaced.
The canonical initialization and three assignment starting points are supported;
new boundary, tie, exhaustive-optimum and move-history checks are included.

At the user's request, this revision has **not been run or verified**: no syntax
check, test suite, experiment, PDF generation or archive validation was executed.
The user will run it using RUN_GUIDE.md. No new numerical results are claimed.

Existing results/, report PDF, submission ZIP and their delivery/provenance
records describe the old source revision. Do not submit them with the rewritten
scripts or copy them into the new result directory. Regenerate artifacts after
running and reviewing the new version.

The four deferred findings remain unfixed: generic short-run reuse, Random's
premature completion checkpoint, archive report-file validation, and regenerated
report environment metadata. Workarounds and manual checks are in RUN_GUIDE.md.

## Historical execution record — 9 November 2026 (previous implementation)

All five required scripts and the full ego-Facebook experiments are complete.
Python 3.12.14; exact library versions are in requirements.txt and the report.
Assignment 1 and .gitignore are unchanged.

## Data and numerical validation

- Official SNAP graph: 4,039 nodes, 88,234 edges, one component; no duplicate
  edges or self-loops were removed. Source/checksums are retained in data/ and
  results/manifest.json.
- Direct opinion iteration versus linear solve: maximum discrepancy below
  1.5e-12 for both profiles (required tolerance 1e-8).
- Two-node analytic checks, three endpoint sweeps, gradients, rank-one caches,
  Woodbury evaluations, baseline definitions, Random endpoint choices, witness
  assignment order and bonus local optimality passed independent small checks.
- Task 1: 3,910 candidate solves per profile; every round matches fast greedy.
  U: 50.47 s fresh versus 5.09 s fast; P: 57.25 s versus 5.93 s.
- All 12 full-graph PGD runs reached the independent reference and assigned every
  resistance to an endpoint. Both objectives were run for both profiles.
- Exact pools: U has 26 nodes (overlap 4); P has 24 nodes (overlap 6). Enumerated
  1,300/20,800 configurations for U at k=2/3 and 1,104/16,192 for P. Independently
  verified winning configurations. Greedy matches these restricted optima.
- Separate non-submodularity and non-supermodularity witnesses were found for
  both profiles, with margins exceeding recorded numerical uncertainty.
- Fast greedy beats both paper baselines at every required budget in U-max,
  P-max and U-min for this seeded experiment. This is an empirical finding.
- All 24 bonus starts were locally optimal in the allowed neighborhood; no
  improvement was found. No broader global guarantee is claimed.

## Artifacts and provenance

The report includes all required tables/plots, derivations, actual commands,
software details, raw-reference comparisons and the corrected-paper citation.
Raw outputs and input arrays remain available under results/. The final archive
includes the five scripts, report, dataset, reproduction instructions and results.

The report is generated from completed real experiments. Temporary synthetic
integration artifacts are outside the repository and absent from the submission.
Presentation-only changes after numerical execution are explicitly recorded in
results/execution_provenance.json; numerical scripts and driver logic were
verified unchanged. The final delivery hash matches the packaged source/data.

See results/delivery_audit.json for artifact and extraction validation. Git history
records final delivery; no repository visibility change or force push is needed.
