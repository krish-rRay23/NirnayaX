"""Tests for the runbook catalog builder."""

from __future__ import annotations

import re

from nirnayax.data.runbooks import build_runbooks
from nirnayax.domain.models import RUNBOOK_ID_PATTERN
from nirnayax.domain.taxonomy import Subcategory, all_subcategories, category_of


def test_build_is_deterministic() -> None:
    assert build_runbooks() == build_runbooks()


def test_one_runbook_per_subcategory() -> None:
    runbooks = build_runbooks()
    covered = {rb.subcategory for rb in runbooks}
    assert covered == set(all_subcategories())
    assert len(runbooks) == len(set(Subcategory))


def test_runbook_ids_are_unique_and_well_formed() -> None:
    runbooks = build_runbooks()
    ids = [rb.runbook_id for rb in runbooks]
    assert len(ids) == len(set(ids))
    for rb in runbooks:
        assert re.match(RUNBOOK_ID_PATTERN, rb.runbook_id)
        assert rb.runbook_id.split("-")[1] == rb.category.value


def test_related_runbook_ids_resolve() -> None:
    runbooks = build_runbooks()
    known = {rb.runbook_id for rb in runbooks}
    for rb in runbooks:
        for ref in rb.related_runbook_ids:
            assert ref in known
            assert ref != rb.runbook_id


def test_runbook_category_matches_subcategory() -> None:
    for rb in build_runbooks():
        assert rb.category == category_of(rb.subcategory)


def test_steps_are_ordered_from_one() -> None:
    for rb in build_runbooks():
        assert [step.order for step in rb.steps] == list(range(1, len(rb.steps) + 1))
