# Assignment 2: implementation status

Scope: numerical/workflow correctness and assignment/TA implementation scope.
Source of truth: assignment PDF and `TA_GUIDE.md`. Working branch: `main`.

**Implemented; runtime unverified.** The agent has not run Python, compilation,
CLI help, tests, experiments, installs, or sync. Only source/diff inspection was
performed. The user will execute `RUN_GUIDE.md`. Items below are not test passes.

| ID | Finding | Implemented change |
| --- | --- | --- |
| C1 | Short/partial worker outputs could be reused as complete | Removed automatic reuse entirely. The driver rejects nonempty output directories and always executes all workers. The required shortlist comparison checks both lengths before all ten rows. |
| C2 | Random checkpoint was complete before aggregates existed | Every trial-only checkpoint has `complete=false`; only the final aggregate result is complete. |
| C3 | Missing provenance could be relabelled with the current hash | No existing experiment directory is accepted for a full run. New runs record their own manifest, arrays, commands and fingerprint without adopting cached workers. |
| C4 | Four local starts instead of three | Exactly `upper`, `initial`, `x_random`, across U/P max/min: 12 runs. Updated producer, required checks, analysis and exports. |
| C5 | Random's memory values were mislabelled as per-budget | Top-level full-process peak; fresh Random processes at k=5 and k=10 plus normal k=30 for the required 36-row comparison table. |
| C6 | Bonus accepted short/incomplete greedy input and mislabelled budgets | Requires completed rounds 1–30 with 30 distinct target indices before producing bonus results. |
| E1 | A second full-graph optimizer was mandatory | Removed Bellman policy iteration and all reference-gap dependencies. Retained the agreed Section 6.1 solve-based search. |
| E2 | Other branches bundle report/archive generation | Kept these features out of main; no branch merge or report automation. |
| E3 | Unused imports/state | Removed unused imports/state and fields/callers belonging to removed verification routines. |

## Agreed implementation boundary

- The corrected paper's solve-based Section 6.1 strategy remains; Algorithm 1
  opinion iteration and PGD are not added. The alpha=1 adaptation uses `s-Pz`.
- Only the assignment's explicit checking experiments remain. Removed optional
  inverse/update verification, independent witness/exhaustive re-solves,
  complement/replay regressions, numerical certificates, and the move-cap option.
- Numerical stopping requires a computed improvement over `1e-9`; final objectives
  from three starts must agree within `1e-5`. These are numerical choices, not
  rigorous certificates or assignment-prescribed thresholds.
- Basic solver failure handling and required budget/cardinality conditions remain.
- No report generator, report refresh, packaging automation or new test suite.
  No file/commit housekeeping issues. A manual report checklist remains deferred.

## Required checks and coverage for the user to execute

| Task | Required behavior retained |
| --- | --- |
| Model/input | Graph counts, preprocessing metadata and original SNAP IDs |
| Task 1 | Three two-node cases, 21-point sweeps on three small graphs, Facebook iteration/solve agreement within `1e-8`, top-200 fresh-solve greedy through k=10 for U/P |
| Task 2 | Fast/naive shortlist comparison, three starts, exhaustive min/max checks on two small graphs, Baselines I/II, 20-trial Random, Example 13, Facebook marginal evidence |
| Task 3 | Top-15 degree/gain union, exhaustive size-2/3 endpoint assignments, optimum IDs/values, pool overlap, counts and time |
| Task 4 | Required objective/runtime/target plots, 36-row main table, 32-row reference table, 12-row local table, increasing marginals and flip/swap bonus |

No additional confirmed Task 1 or Task 3 algorithm defect was identified during
static review. This is not proof that a fresh execution will succeed. User-run
acceptance includes small-graph checks, complete experiments/analysis, refusal
of nonempty full-run directories, and rejection of incomplete bonus inputs.

## Other branches

The two `origin/fix/asgn2-*` references point to the same commit (`9648c2a`),
and `refactor`/`origin/refactor` point to `e92f46d`. Their report generators and
validation frameworks were not imported. Other branches were not modified.
