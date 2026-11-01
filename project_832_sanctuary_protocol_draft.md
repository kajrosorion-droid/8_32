# PROJECT 832 — Sanctuary experiment protocol
**ID:** P832-SANCTUARY-DRAFT-01  
**Version:** 0.1  
**Status:** DRAFT — NOT FROZEN

## Purpose
Test whether a learned representation linking a specific place to benefit changes return behavior beyond direct environmental and somatic effects. This Markdown—not the infographic—is the protocol artifact. Its SHA-256 identifies the exact bytes; it does not prove that code implements the protocol.

## 1. Experimental arms
The four arms below form a 2×2 design (benefit present/absent × explicit place-benefit linkage allowed/blocked). `NO_SANCTUARY` is an additional baseline, not a factorial cell.

| Arm | Structure | Energy/SOMA benefit | Explicit place→benefit linkage |
|---|---|---|---|
| `S_REAL` | Present | Yes | Allowed |
| `GHOST` | Present, visually matched | No | May form |
| `S_REAL_NOMEM` | Present | Yes | Blocked |
| `GHOST_NOMEM` | Present, visually matched | No | Blocked |
| `NO_SANCTUARY` | Absent | No sanctuary-specific benefit | None |

`NOMEM` does not mean all memory is disabled. It blocks only the specified linkage; ordinary spatial memory and unrelated state remain unless separately intervened on.

## 2. Geometry
- `C`: (52,52), radius 8
- `P1`: (20,20), radius 12
- `P2`: (84,84), radius 12

`S_REAL` and `S_REAL_NOMEM` receive the configured energy support and SOMA repair. Ghost arms have the same visible geometry but no sanctuary-specific energy/SOMA effect. `NO_SANCTUARY` has no sanctuary objects.

**Must be specified from code before the main run:** exact benefit/repair rates, capacity, depletion/competition rules, and whether a body fragment or only the designated agent center counts as entry.

## 3. Interventions
For both `*_NOMEM` arms:
- Do not create PN2/PN3 events explicitly encoding benefit for sanctuary coordinates.
- Do not pass `soma_ctx` to `_pn2_tick` / the relevant stance channel.
- Do not update the explicit place→benefit association in ConceptGraph.
- Audit possible leakage through energy, SOMA, gratitude, grief, soul weight, cognitive tension, cellular endurance, stress, and any other state channels.

This is a deep benefit-blind intervention, not a claim that all indirect information is absent. Intent blocking is postponed to round two.

## 4. Entry, exit, return
- Entry: designated agent center is inside a sanctuary circle.
- Exit: center is outside that circle for 2 consecutive ticks.
- Return: after exit, the agent reaches at least `2 × that sanctuary's radius` from its center and re-enters within 100 ticks.
- Link events by agent ID and sanctuary ID.
- Visits lacking a full 100-tick follow-up at run end are right-censored, not counted as failures.

## 5. Metrics and formulas
Let `R(A)` be the seed-level return rate in arm A, using eligible visits with sufficient follow-up. Report counts and exclusions with every rate. Seed—not agent or tick—is the primary independent replication unit.

Primary contrasts:
- Environmental benefit with explicit linkage blocked: `Δ_env = R(S_REAL_NOMEM) − R(GHOST_NOMEM)`
- Added contribution of linkage when benefit exists: `Δ_rep = R(S_REAL) − R(S_REAL_NOMEM)`
- Structure/representation without physical benefit: `Δ_ghost = R(GHOST) − R(GHOST_NOMEM)`
- Total sanctuary contrast vs no-object baseline: `Δ_total = R(S_REAL) − R(NO_SANCTUARY)`
- `R(S_REAL) − R(GHOST_NOMEM)` is the **total** sanctuary contrast, not a pure environmental effect.
- `GHOST − NO_SANCTUARY` is a separate structure/baseline contrast, not a memory effect.

### Outcome-sensitive association (exploratory, not causal)
For each completed visit, define outcome in the 20 ticks after exit (this window must be confirmed before the run):
- `pos_prev`: no energy loss from hunger/damage and no increase in fragmentation.
- `neg_prev`: energy loss from hunger/damage or increased fragmentation.
- Mixed/unobserved: label `mixed/unknown`; do not force into either group.

Do not define positive/negative outcome by whether sanctuary healing occurred inside the sanctuary. Report group counts per arm; if either group is absent or too small, mark the contrast non-estimable.

For each estimable arm `A`, compute:
`S(A) = P(return | pos_prev, A) − P(return | neg_prev, A)`

Report `S(S_REAL)`, `S(S_REAL_NOMEM)`, `S(GHOST)`, and `S(GHOST_NOMEM)`. `S(S_REAL) − S(S_REAL_NOMEM)` is an exploratory association, not a causal policy-attribution estimate. Match/adjust for distance from sanctuary and energy at exit; preregister the method and report balance.

## 6. Ghost detectability
- Use only each agent's **first-ever approach / first entry opportunity**, before any sanctuary entry in that agent's lifetime. Exclude later approaches.
- Use only features from the five ticks before that first approach: `fp_mean`, `fp_var`, `dist_to_center`, energy, fragment count, gratitude, grief, and any valid pre-entry visit history.
- Logistic regression; train on some seeds, test on held-out seeds. Split by seed, never by rows from the same seed.
- Report class balance, confusion matrix, accuracy, and uncertainty. Pre-registered flag: held-out accuracy > 0.65 suggests possible pre-entry distinguishability/leakage requiring investigation; near 0.5 only means this classifier did not distinguish the arms.
- Never use post-entry or post-treatment features.

## 7. Censoring and temporary exclusions
- Death/permanent loss of tracking: censor from that time; report reason and time.
- ID reset: close old track as censored and start a new track; do not silently stitch IDs.
- Feral state: **temporary exclusion of feral ticks only**, not permanent censoring. Resume if the same tracked agent returns from feral state.
- Fragment count ≥3: do not automatically discard the whole agent. Mark intervals and apply a preregistered eligibility rule. That exact rule must be finalized before execution.
- Stationarity >100 ticks: do not automatically censor. Exclude only intervals meeting a preregistered inactivity rule with no target benefit or relevant state change. Long sanctuary occupancy with energy gain or SOMA repair is a target outcome, not a reason for exclusion.
- Report exclusions by arm and reason; never censor based on favorable/unfavorable outcome.

## 8. Seed escalation and stopping
Exploratory stage: 5 seeds/arm, 300 ticks/run. Confirmation stage: 20 seeds/arm. Seed is the replication unit.

Thresholds:
- `S_REAL − GHOST`: 20 percentage points (pp)
- `S_REAL − S_REAL_NOMEM`: 5 pp
- Other contrasts: 20 pp unless changed before data collection.

After 5 seeds, calculate seed-level effect estimates and 95% CIs for each preregistered contrast:
1. Escalate to 20 seeds if the CI includes zero, or if it excludes zero but the absolute effect is below that contrast's threshold.
2. Stop early only if CI excludes zero and absolute effect meets/exceeds its threshold; label this exploratory, not definitive confirmation.
3. 20 seeds is the final planned stage. If still inconclusive, stop and report uncertainty; no post-hoc escalation to 50 without a new protocol.
4. Report effects and intervals, not only significance or pass/fail.

## 9. Interpretation commitments
- If `Δ_rep` is near zero and the interval is narrow enough to exclude effects of practical interest: “The explicit place→benefit linkage showed no detectable contribution beyond tested somatic and energetic signals under this implementation.” Do not generalize to “memory never matters.”
- If `Δ_rep` is near zero but the interval is wide: report “inconclusive/underpowered,” not “no effect.”
- If `S_REAL_NOMEM` returns as often as `S_REAL`, do not claim representation helped. Local energy gradients and somatic state may be sufficient under these conditions, but that is not proof they are the exclusive mechanism.
- A difference between `S_REAL_NOMEM` and `GHOST_NOMEM` supports an environmental-benefit contrast under the specified blocked channels; it does not prove every memory pathway was removed.
- Separate association from causation. Causal attribution requires a later intervention on the information/policy pathway.

## 10. Audit and integrity
Before any main run:
1. Resolve all “must be specified/finalized” items above, including benefit rates/capacity, post-exit window, fragment eligibility, and inactivity rule.
2. Save this file as the canonical protocol.
3. Record its SHA-256, protocol version, code commit, run config, and seed list in `audit.header`.
4. Verify the file hash at run start; abort on mismatch.
5. Log `sanctuary_enter`, `sanctuary_exit`, `sanctuary_intent_formed`, `sanctuary_intent_blocked`, and per-tick sanctuary state to `world.rich.events` (or audited event stream), with agent ID, sanctuary ID, tick, arm, relevant state, and schema version.
6. A hash identifies the document; it does not prove implementation fidelity. Add implementation tests and a code-commit reference.

## 11. Current status
**DRAFT — NOT FROZEN.** This protocol contains proposed definitions; implementation-specific parameters and eligibility rules remain open. Do not start the main experiment until they are resolved and the protocol/code commit are recorded.
