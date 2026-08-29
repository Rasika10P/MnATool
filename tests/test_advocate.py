import pytest

from agents.advocate import _build_human_message, contest_mapping
from agents.negotiation_schemas import AdvocateOutput, CrosswalkArgument
from tests.fakes import FakeModel


def _contest_output(**overrides) -> AdvocateOutput:
    fields = dict(
        argument_basis="scope evidence not reflected in the mapping",
        proposed_level="L7",
        evidence_cited="company-wide final authority on core performance/power/area "
        "tradeoffs across the entire NX-400 through NX-600 roadmap",
        framework_section="nyx_level_framework.md section 4, Distinguished MTS anchor",
    )
    fields.update(overrides)
    return AdvocateOutput(**fields)


def _decline_output() -> AdvocateOutput:
    return AdvocateOutput()


def test_human_message_includes_all_three_inputs():
    message = _build_human_message("Owns X.", "Distinguished MTS", "L6")
    assert "Owns X." in message
    assert "Distinguished MTS" in message
    assert "L6" in message


def test_contest_mapping_returns_contested_argument():
    fake = FakeModel(_contest_output(), schema=AdvocateOutput)
    output = contest_mapping("role summary", "Distinguished MTS", "L6", model=fake)
    assert output.contests is True
    assert output.argument_basis == "scope evidence not reflected in the mapping"
    assert output.proposed_level == "L7"

    argument = output.as_crosswalk_argument()
    assert isinstance(argument, CrosswalkArgument)
    assert argument.proposed_level == "L7"


def test_contest_mapping_returns_decline():
    fake = FakeModel(_decline_output(), schema=AdvocateOutput)
    output = contest_mapping("role summary", "MTS II", "L3", model=fake)
    assert output.contests is False
    assert output.as_crosswalk_argument() is None


@pytest.mark.parametrize(
    "inadmissible_basis",
    ["misapplied factor variant", "misread factor anchor", "Meridian precedent"],
)
def test_contest_mapping_rejects_bases_this_advocate_has_no_access_to(inadmissible_basis):
    # These three bases are schema-admissible in general (they're valid CrosswalkArgument
    # values elsewhere in the negotiation), but this advocate never sees Meridian's own
    # framework or population, so it cannot legitimately support them -- a model that
    # produces one anyway is a model error, and contest_mapping must refuse to pass it through.
    bad_output = _contest_output(argument_basis=inadmissible_basis)
    fake = FakeModel(bad_output, schema=AdvocateOutput)
    with pytest.raises(ValueError, match="no access to Meridian's framework"):
        contest_mapping("role summary", "Distinguished MTS", "L6", model=fake)
