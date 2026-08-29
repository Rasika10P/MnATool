"""Tests for the shared tag-leak detection/cleanup (error_handling_backlog.md entry 1) used
by every schema's field validators and by agents/instrumented_model.py's retry check."""

from agents.text_sanitization import contains_leaked_tags, find_leaked_tag_strings, sanitize_prose_field, strip_leaked_tags


def test_contains_leaked_tags_true_for_a_closing_tag():
    assert contains_leaked_tags("Owns the subsystem end to end. </reasoning>") is True


def test_contains_leaked_tags_true_for_a_tool_call_parameter_fragment():
    assert contains_leaked_tags('<parameter name="alternative_level">L6') is True


def test_contains_leaked_tags_false_for_clean_prose():
    assert contains_leaked_tags("Owns the subsystem end to end, no direct reports.") is False


def test_strip_leaked_tags_removes_the_tag_and_collapses_whitespace():
    dirty = "Senior Staff Engineer). </reasoning>\n<parameter name=\"alternative_level\">L6"
    assert strip_leaked_tags(dirty) == "Senior Staff Engineer). L6"


def test_strip_leaked_tags_is_a_no_op_on_clean_text():
    clean = "Owns the RTL subsystem across two tapeouts."
    assert strip_leaked_tags(clean) == clean


def test_sanitize_prose_field_passes_through_non_strings_unchanged():
    assert sanitize_prose_field(None) is None
    assert sanitize_prose_field(5) == 5


def test_find_leaked_tag_strings_walks_nested_dicts_and_lists():
    args = {
        "evidence_cited": "owns the subsystem </evidence_cited>",
        "framework_section": "section 3",
        "factor_ratings": [{"evidence": "clean"}, {"evidence": '<parameter name="x">'}],
    }
    leaked = find_leaked_tag_strings(args)
    assert "owns the subsystem </evidence_cited>" in leaked
    assert '<parameter name="x">' in leaked
    assert len(leaked) == 2


def test_find_leaked_tag_strings_empty_for_clean_args():
    args = {"evidence_cited": "clean text", "nested": {"a": "also clean"}}
    assert find_leaked_tag_strings(args) == []
