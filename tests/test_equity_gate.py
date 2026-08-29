"""agents/equity_gate.py is deterministic (no model call -- see its module docstring for
why), so these tests run against the real committed data like tests/test_data_access.py
does, rather than a fake model. The L7/engineering population is small and fixed enough to
assert exact figures against, not just loose bounds.
"""

import pytest

from agents.equity_gate import check_equity
from agents.negotiation_schemas import EquityGateResult


def test_check_equity_vetoes_a_candidate_above_every_family_peer():
    # engineering/L7 real population: MER-0234 (PD-STA-L7, US-SJC, compa ~0.99),
    # MER-0235 (SV-TE-L7, EU-MUC, compa ~0.72), MER-0236 (SA-ARCH-L7, IN-BLR, compa ~1.08).
    # A candidate in a fourth geo (Austin, nobody currently at L7 there) priced well above
    # the group's max compa-ratio must fail the gate against all three, demonstrating the
    # cross-geo aggregation actually pulls in incumbents outside the candidate's own geo.
    result = check_equity(
        family_group="engineering", level_code="L7", candidate_geo_code="US-AUS", candidate_salary=280_000.0
    )
    assert isinstance(result, EquityGateResult)
    assert result.passed is False
    assert set(result.conflicting_incumbents) == {"MER-0234", "MER-0235", "MER-0236"}
    assert "1.18" in result.reasoning or "compa-ratio" in result.reasoning


def test_check_equity_passes_a_candidate_below_the_family_max():
    # Same population as above; a modest Austin L7 offer near range_mid (compa ~1.0) sits
    # below MER-0236's ~1.08 and so does not exceed the entire peer group.
    result = check_equity(
        family_group="engineering", level_code="L7", candidate_geo_code="US-AUS", candidate_salary=238_182.11
    )
    assert result.passed is True
    assert result.conflicting_incumbents == []


def test_check_equity_passes_vacuously_with_no_family_peers():
    # corporate/L3 has a salary structure (so the candidate's own compa-ratio is still
    # computable) but zero incumbents at any geo in this dataset -- nothing to be placed
    # above, distinct from corporate/L1, which has no salary structure at all.
    result = check_equity(
        family_group="corporate", level_code="L3", candidate_geo_code="US-SJC", candidate_salary=100_000.0
    )
    assert result.passed is True
    assert result.conflicting_incumbents == []
    assert "No Meridian incumbents" in result.reasoning


def test_check_equity_raises_when_no_salary_structure_exists_for_candidate_geo():
    # corporate/L1 has no salary_structures row at all in this dataset -- the candidate's
    # own compa-ratio can't be computed, so this must surface as an error rather than a
    # silent pass; distinct from the "structure exists, zero incumbents" case above.
    with pytest.raises(ValueError, match="No salary_structures row"):
        check_equity(family_group="corporate", level_code="L1", candidate_geo_code="US-SJC", candidate_salary=100_000.0)
