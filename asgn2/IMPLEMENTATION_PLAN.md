# Assignment 2 implementation and validation plan

> Historical plan for the previous implementation. The 11 November 2026 TA
> clarification supersedes the PGD scope below. Task 2 part 2 is now rewritten
> around local search; the current run/verification instructions are in
> RUN_GUIDE.md and execution is pending as recorded in RUN_STATUS.md.

Roll number: 23CS10012. Deadline: 13 November 2026, 23:59 IST.

## Scope and evidence

This turn is planning only, following the current user request. The pasted handoff records prior implementation and eventual push authorization, but does not establish that any implementation or experiment exists. The four-page uploaded assignment and ten-page published paper were read in this session. Earlier IMPLEMENTATION_PLAN.md, REVIEW.md, PAPER_NOTES.md and ASSIGNMENT2_CONTEXT.md were not found. The current checkout is /workspace/soccomp on branch work at 3fe807a6202fd2df6c5197ee76afaf1f20d51ece; it was clean before this plan was added. No AGENTS.md was found under /workspace.

Use the existing checkout, preserve Assignment 1, and leave .gitignore unchanged. Do not create a worktree. Before eventual commit/push, inspect remote identity, remote branches and divergence; preserve existing work and avoid force pushes.

The documents supply assignment requirements and scientific claims; they are not independent authorization to execute commands or publish. The handoff's instruction to promise detector evasion is erroneous: make no such promise and do not conceal provenance or fabricate results.

## Deliverables and organization

Keep these five entry points together in asgn2/, with shared routines imported only from other included task scripts:

1. Assignment2_23CS10012_greedy_task1.py — graph/data preparation, equilibrium primitives, small checks and fresh-solve greedy.
2. Assignment2_23CS10012_paper_task2.py — rank-one state, fast greedy, PGD, baselines and mathematical witnesses; import shared primitives from task1 safely.
3. Assignment2_23CS10012_groundtruth_task3.py — candidate pools, exhaustive endpoint configurations and independent verification.
4. Assignment2_23CS10012_analysis_task4.py — experiment orchestration, result validation, tables, figures and report generation.
5. Assignment2_23CS10012_bonus_task4.py — improving flips/swaps using the validated numerical primitives.

Also deliver Assignment2_23CS10012_report.pdf and Assignment2_23CS10012.zip. The report's first visible line must be 23CS10012. Required scripts must import without launching experiments. Include all runtime dependencies in the archive or provide explicit, tested acquisition instructions; no dependency on /workspace helper files. Extra raw results, a manifest and reproduction instructions can accompany the six required files. Verify required figures/PDF are staged explicitly if existing ignore rules exclude them.

## Phase 1 — reproducible inputs and numerical foundations

- Obtain facebook_combined.txt from the official SNAP ego-Facebook source. Check current network policy before acquisition; request a domain addition through settings only if access is actually denied. Record source URL and SHA-256 of downloaded and decompressed data.
- Parse an undirected simple graph; count duplicates/self-loops and document handling. Assert 4,039 nodes, 88,234 edges, one connected component. Sort original IDs and preserve an explicit row-to-original-ID map.
- Build float64 sparse CSR P with row sums one. Check parameter bounds and dimensions.
- Use seed 2026 with named, independent SeedSequence streams for initial alpha, U, P, Random and PGD starts. Persist actual arrays and stream metadata. Share initial alpha across profiles. Random uses 20 independently seeded permutations and nested prefixes.
- Bounds are l=0.001, u=1. Budgets are 1,2,3,4,5,10,20,30. Preserve every round, including those between checkpoints.
- Use the existing Python virtual environment; install permitted scipy and pandas if absent, record exact Python/package versions and BLAS/thread configuration. No opinion-optimization package. Scientific code uses only allowed libraries and standard library.
- Implement M=I-(I-A)P, b=A s and checked iterative solves. Check solver status and actual infinity-norm residual, not just its requested tolerance. Use a documented fallback if convergence fails; count recovery solves.
- Because ||M^-1||_infinity <= 1/l, residual r bounds equilibrium error by r/l and total-objective error by n*r/l. Use these bounds or tighter validated bounds when resolving close gains and certifying output precision. Recompute near ties at higher accuracy.
- Define deterministic ties by original node ID, then endpoint order l before u, within a documented numerical tolerance. Use the same rule in slow and fast greedy.

Resolve ties against the best computed objective in the entire candidate batch, rather than pairwise approximate comparisons (which need not be transitive). Use error bounds to identify uncertain contenders, refine them, then apply the fixed ID/endpoint ordering. A claimed positive gain must exceed its numerical uncertainty. Record tolerances explicitly after the small-graph pilot and before production runs; freeze them in the experiment manifest.

Validation gate: derive the two-node solution with D=alpha_a+alpha_b-alpha_a*alpha_b:

z_a=[alpha_a*s_a+(1-alpha_a)*alpha_b*s_b]/D,
z_b=[alpha_b*s_b+(1-alpha_b)*alpha_a*s_a]/D.

Check three parameter choices including boundary/asymmetric cases. On Facebook, compare direct iteration and the solver in infinity norm to 1e-8 for both profiles. On three small graphs (e.g. path, triangle, star, each <=8 nodes), sweep one alpha over 21 evenly spaced values including endpoints. Preserve plots and demonstrate endpoint maximisation; explain coordinate monotonicity/linear-fractional dependence, allowing flat ties. Validate complement identity f_(1-s)=n-f_s.

## Phase 2 — Task 1 and fast greedy

Task 1 uses the 200 highest-degree candidates, with original-ID degree ties, but solves equilibrium on all 4,039 nodes. For both U and P maximisation, run ten rounds, evaluate BOTH endpoints for every remaining candidate by fresh equilibrium computation, and retain earlier assignments. Count 3,910 candidate solves per profile separately from initial, verification and recovery solves. Store added ID, endpoint, objective, gain, solver diagnostics, cumulative solve counts and wall time every round.

For fast greedy let G=M^-1, delta=a_new-alpha_v, g=G[:,v], h=P[v,:]G, d=1+delta*h[v]. Implement:

- z_new = z + delta*(s[v]-(Pz)[v])*g/d
- G_new = G - delta*g*h/d
- delta_f = delta*(s[v]-(Pz)[v])*sum(g)/d

Cache column sums and diag(PG). Update these with the same rank-one change, using P*g for the diagonal update. Do not divide by 1-alpha: alpha=1 must work. Candidate scoring uses scalar cached quantities and no new equilibrium solves. Compute the selected h only when accepting an update. Account for matrix construction and validation overhead.

Specifically, with beta=delta/d, c=1^T G, p=Pz and t=diag(PG), use OLD state values to compute c_new=c-beta*c[v]*h, t_new=t-beta*(P*g)*h, and p_new=p+beta*(s[v]-p[v])*(P*g). Copy views needed by the update before mutating G. The plan's formulas were checked on a seven-node numerical probe, but implementation still requires the full validation gate below.

Benchmark one trustworthy inverse construction on this machine before large runs. Reuse factorizations for block RHS solves where useful; validate M*G against identity using sufficient checks, and validate selected columns independently. One dense 4039-by-4039 float64 array is approximately 124.5 MiB; budget for copies, factorization and outer-product temporaries. Limit BLAS threads and parallel experiments to avoid oversubscription. No runtime estimates are claimed until measured.

Validation gate: small-graph random endpoint updates versus fresh solves; reversals, endpoint-one cases and near-zero changes; objective and residual drift after sequences. Rebuild safely if a denominator or drift check fails and report that cost. Compare every selected ID, endpoint, gain and objective against Task 1 on its identical shortlist through k=10; explain any ties or discrepancies before proceeding.

Then run all-node fast greedy through k=30 for U-max, P-max and U-min. Do not report Task 1 at k=20 or 30.

## Phase 3 — unbudgeted PGD, references and baselines

The uploaded assignment states the transformed objective is convex. Retain the handoff's explicit user decision to execute PGD in X coordinates. Do not invent a convexity proof or convergence agreement. Independently verified counterexample: for a two-node edge with s=(1,0), f(x,y)=(2y-1)/(x+y-1). f(2,2)=1 exceeds the average 7/8 at (2,1),(2,3), and is below the average 9/8 at (1,2),(3,2). Thus the objective is neither convex nor concave. Inspect and cite the corrected paper at arxiv.org/abs/2011.02583 before attributing its precise claims; the current session has read the original paper only.

Set B=X-(X-I)P. Solve Bz=s and B^T q=1. Gradient is -q*((I-P)z), elementwise. Validate central finite differences at interior points on small graphs. Implement projected gradient with backtracking, explicit projected-gradient stopping criteria, residual checks and a recorded iteration cap. Use negative f or complemented opinions for maximisation. Record convergence failure honestly; tiny objective changes alone do not establish convergence.

PGD numerical safeguards: the box spans [1,1000], so a fixed unit step and an absolute gradient cutoff can stall near large x. Permit trial-step growth between iterations, then backtrack with projected Armijo acceptance for the signed objective F: F(x_new)<=F(x)+c_armijo*grad(F)^T*(x_new-x). Record the step, projected-gradient mapping ||x-project(x-eta*grad(F))||_infinity/eta, raw gradient scale, displacement, objective change, and solve residuals. Freeze line-search/stopping parameters after the pilot. Distinguish projected-stationary, reference-certified, iteration-limit and numerical-failure outcomes. Any objective difference used in acceptance must be resolved above solver uncertainty. Do not silently substitute an optimizer or return a polished solution as the PGD output.

Run BOTH maximisation and minimisation for BOTH U and P, each from three recorded starting points (initial 1/alpha, box midpoint, seeded interior random point). This covers the broad wording of Task 2 even though Task 4 requires only U-max, P-max and U-min. Compare objective values, convergence diagnostics and endpoint status. Do not round interior coordinates silently or call an uncertified PGD result an optimum.

On two graphs of <=15 nodes, enumerate every 2^n endpoint assignment and compare with all PGD starts. For independent Facebook reference certification, use the max/min Bellman operator H_i(z)=extreme over a in {l,u} of a*s_i+(1-a)*(Pz)_i. Its contraction factor is at most 1-l; residual r bounds objective error by n*r/l. This validates references, not replaces PGD. If endpoint polishing is used, report it as a separate result. Distinguish theoretical polynomial-time claims from empirical convergence; a fixed-l contraction argument and/or a verified corrected theorem needs a precise accuracy/complexity statement.

Reference construction must also yield an attainable resistance vector: choose endpoint actions from the Bellman iterate, solve that policy independently, and evaluate its Bellman residual. Explain why the Bellman fixed point dominates (for max) or is dominated by (for min) every fixed-policy equilibrium, and why its chosen actions attain it. Report an achievable objective and a bounded gap to the global reference. A small Bellman residual of an intermediate vector alone does not certify a separately reported policy. Bound floating-point evaluation error as well as solver error, and label results numerical certificates unless formal interval arithmetic is used. Reaching an epsilon-accurate value in time polynomial in 1/l is not, by itself, an exact polynomial-time result in the bit length of an arbitrary l. Keep this distinction in the theoretical claim verdict.

Baseline I: rank descending innate s, set selected alpha to 1.
Baseline II: rank descending [degree(i)/(2m)]*[s_i/sum_neighbor(s)], set selected alpha to 1. Do not replace the neighbor sum with an average. Define positive/zero as +infinity, 0/0 as zero, and original-ID ties.
For minimisation rank using 1-s, still assign alpha=1, and evaluate original s.
Random: 20 seeded uniform random permutations; each prefix is a uniform k-set. In its sampled order assign each newly selected node its better endpoint against the current equilibrium, retaining earlier choices. This is the approved sequential convention, not joint endpoint optimization. Report mean, sample variability and individual trials. Use matching candidate universes for restricted comparisons and label all-node results.

## Phase 4 — mathematical witnesses and Task 3 exact search

Reproduce published Example 4 (triangle, s=(1,0,0), alpha=(0.1,0.1,0.1)), explaining the assignment's Example 13 numbering mismatch. Enumerate endpoint assignments for each set. Exact expected objectives: f(empty)=1; f({1})=29/11; f({3})=2999/2009; f({1,2})=f({1,3})=86971/31009; f({1,2,3})=2999/1001. The printed supermodularity marginal 1.493 omits subtracting f(empty)=1; the correct value is approximately 0.493 and the violation remains.

Seek separate explicit Facebook witnesses for non-submodularity and non-supermodularity. For each A subset B and v outside B, compute the SAME set function at all four sets by exhaustive endpoint assignment on small sets. Verify the inequality exceeds numerical uncertainty using fresh solves. Search deterministically and retain IDs, endpoint optima, objectives and margins. Order-dependent greedy histories are not proofs about the optimized set function. If one witness is not found, distinguish theory from what was demonstrated on this instance.

For U-max and P-max separately, build C as union of top-15 degree and top-15 singleton best-endpoint gains. Save both lists, overlap, IDs and |C|. Enumerate all size-2 and size-3 subsets, then all 2^k endpoint assignments; count choose(|C|,k)*2^k configurations. Reuse the original system with Woodbury/small dense solves, not new full inverses. Independently solve each winning configuration freshly. Report target IDs, endpoints, objective, counts and time. T* is exact only within C. Run greedy restricted to C. All-node methods can exceed this restricted reference; do not clip gain ratios above one.

Concrete simultaneous-update evaluator: relative to a fixed base M,G,z, let S be the changed indices, E the columns of identity indexed by S, R=P[S,:], D=diag(alpha_new[S]-alpha_base[S]), H_S=R*G*E and c_S=1^T*G*E. Solve the small system (I+D*H_S)*w=D*(s[S]-R*z); then z_new=z+G*E*w and f_new=f_base+c_S*w. This includes zero changes without dividing by D. Do not accidentally transpose H_S or reverse the order of D and H_S. Validate full vectors on small graphs before using objective-only evaluations.

For |C|=30 expect 1,740 k=2 configurations and 32,480 k=3 configurations, with separate subset/configuration counters. Inspect conditioning and residuals of the small systems. Check all numerically competitive configurations independently, not only the apparent winner; keep a best-versus-runner-up gap and an uncertainty estimate. Use a canonical target-set/endpoint tie convention and describe numerically indistinguishable optima. Jaccard overlaps are relative to that chosen reference. Exhaustive enumeration establishes combinatorial coverage; ordinary float64 evaluation alone is not a formal exact-arithmetic certificate.

Performance gate for Random and bonus: avoid a dense 4039-by-4039 inverse update for each trial prefix or rejected swap. Random can use the simultaneous-update evaluator on a fixed base for sets of size <=30, testing the two endpoints of each new node while retaining accepted earlier choices. Bonus can score a flip with rank one and a removal/insertion with rank two against its current state; mutate the large state only for accepted moves. Validate these paths against fresh solves. Reusing a base inverse across experiments is allowed for execution, but report its amortization explicitly and charge each standalone algorithm its required setup in fair comparisons.

## Phase 5 — analysis and bonus

For U-max, P-max and U-min produce objective-versus-budget curves for fast greedy, I, II and Random with initial and certified unbudgeted-reference lines. Clearly label any uncertified reference. Include Random in all applicable method comparisons, with matching pool labels and uncertainty.

Plot cumulative Task 1 versus matching-shortlist fast-greedy runtimes through ten; show all-node fast-greedy timing separately. Include initialization costs rather than comparing only update kernels. Measure each algorithm/scenario in a separate subprocess with Linux resource.getrusage peak RSS (KiB converted explicitly); report total process peak, not Python allocations alone. To report peak at k=5,10,30, capture the high-water mark at each checkpoint. Record hardware, threads, solver counts, setup and validation timing. Random's aggregate 20-trial cost and per-trial costs must be distinguishable.

Required table: f, fraction of unbudgeted improvement, cumulative time and peak memory at k=5,10,30. Positive improvement is f-f0 for max and f0-f for min. Treat zero denominators as undefined with an explanation.

At k=2,3, compare greedy and every baseline with T*: improvement ratios, Jaccard overlap, and restricted-C greedy. For Random compute per-trial ratios/overlaps and summarize them; a mean objective does not define a target set. Explain restricted versus unrestricted universes.

These T* comparisons apply to U-max and P-max only, because Task 3 defines no minimisation reference. Show the required all-node methods against C's optimum and a clearly separated within-C comparison for greedy, Baselines I/II and Random. Keep different pools' Random permutations distinct and recorded. At k=5,10,30 the main table has all four methods in each of the three required scenarios; use the same reference and initial objective within each scenario.

Scatter innate s_i against pre-intervention mean equilibrium neighbor opinion (Pz0)_i for greedy targets, marking endpoint and scenario. Preserve all 30 rounds and count increases in consecutive positive marginal improvements above numerical tolerance.

Bonus: at each requested scenario/budget initialize from fast greedy, retain cardinality, and repeatedly scan endpoint flips and swaps with currently unselected nodes from the 200-degree shortlist. When removing a target restore its ORIGINAL alpha, and test both endpoints of the inserted node. Use deterministic best-improvement selection, original-ID ties, and a declared improvement tolerance; rescan until no allowed move improves. Validate accepted moves and final local optimality. Report gains, scan counts, time, proportion of starts already locally optimal and a comparison plot. Zero added improvement is a valid outcome.

Give evidence-based verdicts for all four claims in Task 2. Matching three PGD runs is not a proof of global optimality; one diminishing-returns violation is not also a supermodularity violation; superior greedy at some budgets is not superiority at every budget.

## Phase 6 — reproducibility, report and delivery

Use checkpointed JSON/CSV/NPZ results with input/config hashes, original IDs, seeds, exact versions, commands, tolerances, stop reasons, timings and memory. Resume only matching configurations; never silently mix stale runs. Save every required result before generating report figures.

Launch measurement workers using a fresh interpreter (subprocess/exec), not a fork-only child inheriting a parent that already allocated dense arrays. Prevent BLAS oversubscription before imports. For resumed runs, retain completed segment times, report restart overhead separately, and use the maximum recorded per-segment peak RSS; do not present restarted execution as a single uninterrupted benchmark. Distinguish cold setup from explicitly amortized setup. Random memory is the maximum worker peak, not the sum across sequential trials; report trial-time dispersion and aggregate time separately.

The report begins with roll number and contains software requirements, data preprocessing, equilibrium derivation, rank-one and gradient derivations, bounds/seeds/commands, all required tables/plots, baseline conventions, discrepancies/corrections, limitations and references. Use a report-generation path that can be reproduced with permitted dependencies (for example Matplotlib PdfPages for text/table/plot pages), and visually inspect pagination, labels, equations and the first line.

Audit each rubric item and exact filename. Extract the archive into a clean temporary directory; run entry-point help/import checks, the complete small-graph validation suite and a representative dataset command using only archived resources and documented dependencies. Verify archive contents, data/checksum instructions and result-to-figure provenance. Do not regenerate all expensive experiments merely to check extraction.

Review Git changes; stage only Assignment 2 deliverables and chosen reproducibility artifacts, preserving Assignment 1 and existing ignore rules. Use the existing authorized delivery scope for commit/push after successful implementation and review; verify actual remote success and report the commit ID. This planning turn creates no assignment implementation, report, archive, commit or push.

## Execution order and completion gates

1. Inputs, deterministic configuration, small numerical checks and gradient validation.
2. One-round performance pilot; choose matrix setup and resource settings from measurements.
3. Complete fresh-solve shortlist runs and matching fast-greedy validation.
4. Full fast greedy, baselines, Random, PGD/reference checks, witnesses and exhaustive restricted search.
5. Bonus local search and all analysis outputs.
6. Report, rubric audit, clean archive validation, Git review and delivery.

Do not advance past unexplained numerical mismatches. Persist partial experiments so an environment interruption does not require repeating completed work. No substantive user clarification is currently required: the handoff resolves Random, baseline assignment, Example numbering and keeping PGD. Any later instructor reply can change the PGD interpretation without invalidating independent completed work.
