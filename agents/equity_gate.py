"""The equity agent (level_framework.md section 7, "Participants"). Runs only when the
arbiter's verdict is "revised" -- the last check before a revision is final, per section 7:
"Every revision passes to the equity agent before it is final." Not called by this module
itself on any other verdict; the calling convention (only invoke on a revised ruling) is
enforced by whatever wires the negotiation subgraph together, the same way arbiter.rule()
doesn't itself enforce the two-round limit.

Two design decisions worth surfacing rather than leaving implicit:

**Peer group.** check_internal_equity alone is scoped to one exact (job_id, geo_code) --
one sub-family, one location. Decided with the user rather than assumed: at L7 this dataset
has exactly 3 incumbents company-wide, each in a different sub-family *and* a different geo
(Timing/US-SJC, Test Engineering/EU-MUC, Systems Architecture/IN-BLR), so an exact-match
query essentially never finds more than one peer at senior levels even when the family
genuinely has several people at that level. The chosen peer group is the whole family_group
at the proposed level, across every geo -- gathered by calling check_internal_equity once
per (job_id, geo_code) pair that has an incumbent there (tools.data_access.
list_family_level_incumbent_locations), then comparing everyone by compa-ratio rather than
raw salary, since compa-ratio (salary / that person's own geo's range_mid) is already
geo-normalized -- that's the entire reason compa-ratio exists as a metric, so a peer in a
different geo than the candidate is legitimately comparable on this basis even though their
absolute pay isn't. See learnings.md for this decision's record.

**"Demonstrably greater scope," operationalized.** The framework's veto condition names two
things: the revision pays the candidate above incumbents, AND those incumbents have
demonstrably greater scope. This system has no persisted scope narrative for existing
Meridian incumbents to compare against (unlike the Nyx census's role summaries) -- an
incumbent already holding a level got there through Meridian's own leveling process, which
is itself the only evidence of scope this system has for them. So "demonstrably greater
scope" collapses, given what's actually computable here, to "already holds the level" --
this gate's real, computable question is narrower than the framework text reads in
isolation: not "is this specific incumbent's scope greater," but "would the revision pay the
acquired employee more than the entire set of people this system has already validated at
this level." A comp-domain simplification forced by the data model, not an engineering
choice -- flagged, not silently assumed, and worth the comp manager's eyes if this gate ever
starts vetoing more than expected.

Given both of those, the actual pass/fail determination is a fully mechanical threshold
comparison with no open judgment call left in it -- so, unlike the advocate and arbiter,
this module makes no model call. It's deterministic top to bottom, consistent with CLAUDE.md
non-negotiable 1 ("math in code, judgment in agents"): there's no judgment left to put
anywhere once the two decisions above are made. check_equity still returns a validated
EquityGateResult like every other negotiation output, so callers don't need to know or care
that this particular "agent" happens to have no model behind it.
"""

from __future__ import annotations

from agents.negotiation_schemas import EquityGateResult
from tools.comp_math import compute_pay_metrics
from tools.data_access import (
    check_internal_equity,
    list_family_level_incumbent_locations,
    lookup_salary_structure,
)


def _gather_family_level_peers(
    family_group: str, level_code: str, candidate_geo_code: str, candidate_salary: float
) -> dict:
    """Deterministic aggregation, no judgment: the candidate's own compa-ratio against their
    own geo's range, plus every peer's compa-ratio gathered by calling check_internal_equity
    once per (job_id, geo_code) pair with an incumbent in this family_group/level_code. The
    0.0 candidate_salary passed to each check_internal_equity call is a placeholder -- those
    calls are only used for their peer_compa_ratios/source_rows; the real candidate
    comparison happens once, below, against the candidate's own geo."""
    structure = lookup_salary_structure(family_group, level_code, candidate_geo_code)["structure"]
    candidate_metrics = compute_pay_metrics(
        candidate_salary, structure["range_min"], structure["range_mid"], structure["range_max"]
    )

    peers: list[dict] = []
    for job_id, geo_code in list_family_level_incumbent_locations(family_group, level_code):
        result = check_internal_equity(job_id=job_id, geo_code=geo_code, candidate_salary=0.0)
        for row, peer_compa_ratio in zip(result["source_rows"], result["peer_compa_ratios"]):
            peers.append(
                {
                    "employee_id": row["employee_id"],
                    "job_id": job_id,
                    "geo_code": geo_code,
                    "base_salary": row["base_salary"],
                    "compa_ratio": peer_compa_ratio,
                }
            )

    return {
        "inputs": {
            "family_group": family_group,
            "level_code": level_code,
            "candidate_geo_code": candidate_geo_code,
            "candidate_salary": candidate_salary,
        },
        "candidate_compa_ratio": candidate_metrics["compa_ratio"],
        "peers": peers,
    }


def check_equity(
    family_group: str,
    level_code: str,
    candidate_geo_code: str,
    candidate_salary: float,
) -> EquityGateResult:
    """The gate. Fails only when the candidate's compa-ratio exceeds every peer's -- i.e.
    the revision would place the candidate above the entire existing peer group, not merely
    above some individual peer, matching check_internal_equity's own "above_all_peers"
    concept generalized across the family. Passes vacuously when the family has no
    incumbents at this level anywhere -- there's no peer group to be placed above."""
    aggregated = _gather_family_level_peers(family_group, level_code, candidate_geo_code, candidate_salary)
    candidate_compa_ratio = aggregated["candidate_compa_ratio"]
    peers = aggregated["peers"]

    if not peers:
        return EquityGateResult(
            passed=True,
            conflicting_incumbents=[],
            reasoning=(
                f"No Meridian incumbents at {level_code} in {family_group} (any geo) to "
                "compare against -- nothing to be placed above."
            ),
        )

    max_peer_compa_ratio = max(p["compa_ratio"] for p in peers)

    if candidate_compa_ratio > max_peer_compa_ratio:
        conflicting = [p["employee_id"] for p in peers]
        peer_detail = "; ".join(f"{p['employee_id']} ({p['job_id']}, {p['geo_code']}, compa {p['compa_ratio']:.2f})" for p in peers)
        return EquityGateResult(
            passed=False,
            conflicting_incumbents=conflicting,
            reasoning=(
                f"Candidate compa-ratio {candidate_compa_ratio:.2f} exceeds every existing "
                f"{level_code} incumbent in {family_group} (max peer compa-ratio "
                f"{max_peer_compa_ratio:.2f}): {peer_detail}. Revision rejected per section 7's "
                "equity gate -- these incumbents already hold this level through Meridian's own "
                "leveling process, so paying the candidate above all of them fails the gate."
            ),
        )

    peer_detail = "; ".join(f"{p['employee_id']} (compa {p['compa_ratio']:.2f})" for p in peers)
    return EquityGateResult(
        passed=True,
        conflicting_incumbents=[],
        reasoning=(
            f"Candidate compa-ratio {candidate_compa_ratio:.2f} does not exceed every existing "
            f"{level_code} incumbent in {family_group} (max peer compa-ratio "
            f"{max_peer_compa_ratio:.2f} among {len(peers)} peer(s): {peer_detail}). Gate passes."
        ),
    )
