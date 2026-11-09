# Completed execution — 9 November 2026

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
