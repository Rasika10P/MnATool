"""Shared detection and cleanup for one specific structured-output failure mode
(error_handling_backlog.md entry 1): the model's tool call leaking tag-like syntax into a
free-text field -- a stray closing tag, or a fragment of the tool-call protocol itself
(`<parameter name="...">`, `</invoke>`). Observed leaking into LevelingDecision.reasoning and,
worse, into AdvocateOutput.evidence_cited where it rendered directly in the Streamlit page.

One shared regex, used in two independent places for two different reasons:

- agents/instrumented_model.py's retry loop calls find_leaked_tag_strings against the RAW
  tool-call arguments (before Pydantic ever parses them) to decide whether an attempt that
  otherwise validated cleanly should still be retried. A leak can corrupt a field without
  ever raising a ValidationError -- entry 1's own example: the tag ate the JSON structure and
  alternative_level came back null instead of raising -- so this check has to run
  independently of whether parsing itself succeeded, and before any field validator gets a
  chance to clean the value up.
- Every prose field across agents/schemas.py, agents/negotiation_schemas.py and
  agents/modeling_schemas.py runs strip_leaked_tags as a field validator, unconditionally --
  a second, independent guarantee that no leaked tag fragment can reach the UI even on an
  attempt where detection didn't trigger a retry (attempts exhausted, or a leak shape this
  regex's heuristic doesn't catch).
"""

from __future__ import annotations

import re
from typing import Any

# A closing/self-describing XML-ish tag (</reasoning>, </invoke>) or a tool-call parameter
# fragment (<parameter name="...">). Deliberately broad rather than an exhaustive list of
# exact strings observed so far -- entry 4's own conclusion, after seeing three different
# wrapper keys across three attempts on one case, was that the failure mode is "the model's
# tool call doesn't exactly match the schema," not any one specific shape of it. Trade-off:
# this also matches a genuine `<T>`-style generic-type mention (e.g. "templates like
# vector<int>") if one ever appears in prose quoting a job description. Accepted deliberately
# -- a false-positive strip of a benign angle-bracket fragment is far cheaper than leaked
# protocol markup rendering in the UI, and this domain's prose (leveling/negotiation/cost
# reasoning) has little reason to quote code syntax in the first place.
_TAG_LEAK_PATTERN = re.compile(r"</?[a-zA-Z_][a-zA-Z0-9_]*(?:\s+[a-zA-Z_][a-zA-Z0-9_]*=\"[^\"]*\")*\s*/?>")


def contains_leaked_tags(text: str) -> bool:
    """True if `text` contains tag-like syntax that has no business in prose."""
    return bool(_TAG_LEAK_PATTERN.search(text))


def strip_leaked_tags(text: str) -> str:
    """Removes tag-like syntax from `text`, collapsing the whitespace left behind. Safe on
    legitimate prose -- the pattern only matches `<...>`-shaped syntax, which no genuine
    leveling/negotiation/cost reasoning has any reason to contain."""
    cleaned = _TAG_LEAK_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def sanitize_prose_field(value: Any) -> Any:
    """The actual field-validator body every schema below calls: strip if it's a string,
    pass through unchanged otherwise (None, or any other type mode="before" might see before
    Pydantic's own type coercion runs)."""
    return strip_leaked_tags(value) if isinstance(value, str) else value


def find_leaked_tag_strings(value: Any) -> list[str]:
    """Recursively walks a raw tool-call arguments structure (nested dict/list/str, as
    LangChain hands back in a ToolCall's `args` before Pydantic ever parses it) and returns
    every string value containing a leak -- not just a bool, so a leak buried in a nested
    arguments dict is still locatable for logging."""
    found: list[str] = []
    if isinstance(value, str):
        if contains_leaked_tags(value):
            found.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            found.extend(find_leaked_tag_strings(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(find_leaked_tag_strings(v))
    return found
