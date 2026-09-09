"""Tests for the deterministic incident generator and template catalog."""

from __future__ import annotations

import string

import pytest

from nirnayax.data import catalog
from nirnayax.data.generator import (
    DEFAULT_EVAL_SEED,
    DEFAULT_TRAIN_SEED,
    EVAL_ID_START,
    REFERENCE_TIME,
    TRAIN_ID_START,
    generate_dataset,
    generate_incidents,
)
from nirnayax.domain.models import DatasetSplit
from nirnayax.domain.taxonomy import Subcategory, category_of


def _placeholders(template: str) -> set[str]:
    return {field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name}


def test_generation_is_deterministic() -> None:
    assert generate_incidents(250, seed=42) == generate_incidents(250, seed=42)


def test_different_seeds_differ() -> None:
    assert generate_incidents(250, seed=1) != generate_incidents(250, seed=2)


def test_ids_are_sequential_and_unique() -> None:
    incidents = generate_incidents(50, seed=7)
    assert [i.incident_id for i in incidents] == [f"INC-{n:06d}" for n in range(1, 51)]
    assert len({i.incident_id for i in incidents}) == 50


def test_id_start_offsets_the_sequence() -> None:
    incidents = generate_incidents(3, seed=7, id_start=EVAL_ID_START)
    assert [i.incident_id for i in incidents] == [f"INC-{EVAL_ID_START + n:06d}" for n in range(3)]


def test_id_start_shifts_ids_but_not_content() -> None:
    """id_start renames records without disturbing the RNG-driven payload."""

    base = generate_incidents(20, seed=7)
    shifted = generate_incidents(20, seed=7, id_start=EVAL_ID_START)
    for original, moved in zip(base, shifted, strict=True):
        assert original.model_dump(exclude={"incident_id"}) == moved.model_dump(
            exclude={"incident_id"}
        )


def test_train_and_eval_id_spaces_never_overlap() -> None:
    train = generate_dataset(600, split=DatasetSplit.TRAIN)
    eval_ds = generate_dataset(200, split=DatasetSplit.EVAL)
    train_ids = {i.incident_id for i in train.incidents}
    eval_ids = {i.incident_id for i in eval_ds.incidents}
    assert train_ids.isdisjoint(eval_ids)
    assert min(i.incident_id for i in train.incidents) == f"INC-{TRAIN_ID_START:06d}"
    assert min(i.incident_id for i in eval_ds.incidents) == f"INC-{EVAL_ID_START:06d}"


def test_id_start_must_be_positive() -> None:
    with pytest.raises(ValueError):
        generate_incidents(1, seed=0, id_start=0)


def test_ids_exceeding_six_digits_raise() -> None:
    with pytest.raises(ValueError):
        generate_incidents(2, seed=0, id_start=999_999)


def test_every_incident_is_taxonomy_consistent() -> None:
    for inc in generate_incidents(500, seed=3):
        assert inc.category == category_of(inc.subcategory)


def test_reported_at_never_after_reference_time() -> None:
    for inc in generate_incidents(300, seed=9):
        assert inc.reported_at <= REFERENCE_TIME


def test_negative_size_raises() -> None:
    with pytest.raises(ValueError):
        generate_incidents(-1, seed=0)


def test_zero_size_is_empty() -> None:
    assert generate_incidents(0, seed=0) == ()


def test_generate_dataset_uses_split_default_seeds() -> None:
    train = generate_dataset(10, split=DatasetSplit.TRAIN)
    eval_ds = generate_dataset(10, split=DatasetSplit.EVAL)
    assert train.metadata.seed == DEFAULT_TRAIN_SEED
    assert eval_ds.metadata.seed == DEFAULT_EVAL_SEED
    assert train.metadata.split is DatasetSplit.TRAIN
    assert train.incidents != eval_ds.incidents  # different splits never share records


def test_generate_dataset_metadata_is_consistent() -> None:
    ds = generate_dataset(15, split=DatasetSplit.TRAIN)
    assert ds.metadata.size == len(ds.incidents) == 15
    assert ds.metadata.generated_at == REFERENCE_TIME


@pytest.mark.parametrize("subcategory", list(Subcategory))
def test_every_subcategory_has_a_template(subcategory: Subcategory) -> None:
    assert subcategory in catalog.SUBCATEGORY_TEMPLATES


@pytest.mark.parametrize("subcategory", list(Subcategory))
def test_template_placeholders_are_resolvable(subcategory: Subcategory) -> None:
    """Every placeholder used must be a base placeholder or one of the signals."""

    template = catalog.SUBCATEGORY_TEMPLATES[subcategory]
    allowed = set(catalog.BASE_PLACEHOLDERS) | {s.name for s in template.signals}
    for text in template.titles + template.descriptions:
        assert _placeholders(text) <= allowed, (subcategory, _placeholders(text) - allowed)


def test_sampled_signal_values_are_within_spec_ranges() -> None:
    incidents = generate_incidents(600, seed=11)
    for inc in incidents:
        specs = {s.name: s for s in catalog.SUBCATEGORY_TEMPLATES[inc.subcategory].signals}
        for signal in inc.signals:
            spec = specs[signal.name]
            assert spec.low <= signal.value <= spec.high
