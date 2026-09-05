# Leveling eval results

5 of 20 cases labeled.

- **Exact-level match rate:** 100%
- **Within-one-level rate:** 100%
- **Escalation precision:** 50%
- **Escalation recall:** 100%

### Accuracy by family group

| Bucket | Exact-match rate | n |
|---|---|---|
| engineering | 100% | 5 |

### Accuracy by track

| Bucket | Exact-match rate | n |
|---|---|---|
| IC | 100% | 3 |
| MGR | 100% | 2 |

### Accuracy by source

| Bucket | Exact-match rate | n |
|---|---|---|
| adversarial | 100% | 2 |
| synthetic | 100% | 3 |

### Accuracy by source_type

| Bucket | Exact-match rate | n |
|---|---|---|
| internal (no source org) | 100% | 3 |
| whole company | 100% | 2 |

### Per-rule compliance

| Rule under test | n | Cited correctly | Landed on expected level |
|---|---|---|---|
| rule 3: deep-but-narrow does not reach L6 | 1 | 100% | 100% |
| rule 6: title in the source document is evidence, not input | 1 | 100% | 100% |
| section 6 rule 3: platform dependency must be assessed for carve-outs | 1 | 100% | 100% |

### Per-case detail

| Case | Source | Expected | Assigned | Outcome | Escalate (expected/actual) | Governing rule | Rule matched |
|---|---|---|---|---|---|---|---|
| case-01 | synthetic | L4 | L4 | exact | False / False | rule 1: scope of impact is primary and confirms L4 across all factors | — |
| case-02 | synthetic | M3 | M3 | exact | False / False | Manager level determined by scope of impact and problem complexity, corroborated by span & budget (section 5, rule 5); factor 5a applied as this is an engineering family (Embedded Software). | — |
| case-03 | synthetic | L4 | L4 | exact | True / True | rule 6: title in source document is evidence, not input; level from described scope. Also section 6 rule 3: platform dependency caps technical depth/breadth. | yes |
| case-04 | adversarial | M3 | M3 | exact | False / False | rule 6: title in the source document is evidence, not input -- level from described scope, never title | yes |
| case-05 | adversarial | L5 | L5 | exact | False / True | rule 3: deep-but-narrow technical work caps at L5 despite scope/influence/ownership evidence pointing higher | yes |
| case-06 | census | — | L1 | not yet labeled | — / False | rule 2: lower level governs a split | — |
| case-07 | census | — | L4 | not yet labeled | — / True | rule 3: deep-but-narrow technical work caps at L5, but here the evidence itself (single subsystem ownership, informal peer influence, no cross-domain breadth) converges on L4 rather than needing the cap invoked | — |
| case-08 | census | — | L3 | not yet labeled | — / True | rule 2: lower level governs a split (L3/L4 split resolved to L3 since scope of impact is not unambiguously at L4) | — |
| case-09 | census | — | M3 | not yet labeled | — / False | rule 5: manager level is not automatic from headcount -- span corroborates but scope of impact and problem complexity (both M3-level here) govern the assignment | — |
| case-10 | census | — | L5 | not yet labeled | — / True | rule 3: deep-but-narrow technical work caps at L5 | — |
| case-11 | census | — | L2 | not yet labeled | — / True | rule 2: lower level governs a split (scope of impact not unambiguously at the higher level) | — |
| case-12 | census | — | L3 | not yet labeled | — / True | rule 2: lower level governs a split (all factors converge around L3, with no unambiguous higher-level scope evidence) | — |
| case-13 | census | — | L4 | not yet labeled | — / True | rule 2: lower level governs a split (scope of impact not unambiguously at the higher level) | — |
| case-14 | census | — | L4 | not yet labeled | — / True | rule 6: title in the source document is evidence, not input -- level from described scope, never from the title | — |
| case-15 | census | — | M3 | not yet labeled | — / True | rule 2: lower level governs a split (scope of impact not unambiguously at the higher level) | — |
| case-16 | census | — | L6 | not yet labeled | — / True | rule 4: external recognition required for L7+; internal reputation only caps this at L6 despite scope evidence otherwise suggesting L7 | — |
| case-17 | census | — | L3 | not yet labeled | — / True | rule 2: lower level governs a split (though here factors are consistently at L3, not genuinely split) | — |
| case-18 | census | — | L3 | not yet labeled | — / True | rule 2: lower level governs a split | — |
| case-19 | census | — | L4 | not yet labeled | — / True | rule 2: lower level governs a split (with rule 3 as supporting constraint on depth-only work) | — |
| case-20 | census | — | L6 | not yet labeled | — / True | rule 3: deep-but-narrow technical depth caps at L5, but scope of impact, ownership, and influence are unambiguously at L6, so per rule 2 exception scope of impact governs the overall level even though technical depth/breadth alone caps at L5 | — |
