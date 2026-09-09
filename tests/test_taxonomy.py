"""Tests for the incident taxonomy and its integrity guarantees."""

from __future__ import annotations

import pytest

from nirnayax.domain.taxonomy import (
    TAXONOMY,
    Category,
    Subcategory,
    all_subcategories,
    category_of,
    subcategories_of,
)


def test_taxonomy_covers_every_category() -> None:
    assert set(TAXONOMY) == set(Category)


def test_every_subcategory_is_mapped_exactly_once() -> None:
    mapped = [sub for subs in TAXONOMY.values() for sub in subs]
    assert sorted(mapped, key=lambda s: s.value) == sorted(Subcategory, key=lambda s: s.value)
    assert len(mapped) == len(set(mapped))  # no duplicates


def test_required_application_db_subcategories_present() -> None:
    required = {
        Subcategory.CONNECTION_POOL_EXHAUSTION,
        Subcategory.REPLICATION_LAG,
        Subcategory.DEADLOCK,
        Subcategory.SLOW_QUERY,
        Subcategory.DISK_SPACE,
    }
    assert required <= set(subcategories_of(Category.APPLICATION_DB))


@pytest.mark.parametrize("subcategory", list(Subcategory))
def test_category_of_round_trips(subcategory: Subcategory) -> None:
    parent = category_of(subcategory)
    assert subcategory in subcategories_of(parent)


def test_all_subcategories_matches_enum() -> None:
    assert set(all_subcategories()) == set(Subcategory)
    assert len(all_subcategories()) == len(set(Subcategory))
