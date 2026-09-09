"""Tests for the EDA computation."""

from __future__ import annotations

from nirnayax.data.eda import compute_eda
from nirnayax.data.generator import generate_incidents
from nirnayax.domain.taxonomy import category_of


def test_empty_dataset() -> None:
    report = compute_eda(())
    assert report.total == 0
    assert report.by_category == {}
    assert report.description_length is None
    assert "empty" in report.render()


def test_counts_are_consistent() -> None:
    incidents = generate_incidents(500, seed=5)
    report = compute_eda(incidents)

    assert report.total == 500
    assert sum(report.by_category.values()) == 500
    assert sum(report.by_subcategory.values()) == 500
    assert sum(report.by_severity.values()) == 500
    assert sum(report.by_channel.values()) == 500


def test_category_counts_match_manual_tally() -> None:
    incidents = generate_incidents(300, seed=6)
    report = compute_eda(incidents)
    expected: dict[str, int] = {}
    for inc in incidents:
        key = category_of(inc.subcategory).value
        expected[key] = expected.get(key, 0) + 1
    assert report.by_category == expected


def test_customer_impacting_percentage() -> None:
    incidents = generate_incidents(200, seed=8)
    report = compute_eda(incidents)
    manual = sum(1 for inc in incidents if inc.customer_impacting)
    assert report.customer_impacting == manual
    assert report.customer_impacting_pct == round(100.0 * manual / 200, 2)


def test_render_contains_sections() -> None:
    rendered = compute_eda(generate_incidents(100, seed=2)).render()
    for section in ("by category", "by subcategory", "by severity", "by channel"):
        assert section in rendered


def test_description_length_bounds() -> None:
    incidents = generate_incidents(100, seed=4)
    report = compute_eda(incidents)
    assert report.description_length is not None
    lengths = [len(inc.description) for inc in incidents]
    assert report.description_length.min == min(lengths)
    assert report.description_length.max == max(lengths)
