"""Tests for the dedicated preparation benchmark workload."""

import copy
import itertools

import pytest
from ommx import ProvenanceKind, SpecialConstraintKind, State

from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter

from benchmarks.instance import build_one_hot_preparation_instance


@pytest.mark.parametrize(
    ("special_constraints", "expected_kinds"),
    [
        ("indicator", {ProvenanceKind.IndicatorConstraint}),
        ("sos1", {ProvenanceKind.Sos1Constraint}),
        (
            "indicator-sos1",
            {
                ProvenanceKind.IndicatorConstraint,
                ProvenanceKind.Sos1Constraint,
            },
        ),
    ],
)
def test_one_hot_preparation_cases(special_constraints, expected_kinds):
    size = 3
    source = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
    )
    before = source.to_v2_bytes()
    input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS
    assert input_class is not None
    assert not input_class.contains(source)

    prepared = copy.copy(source)
    prepared.prepare(
        input_class,
        OMMXFixstarsAmplifyAdapter.recommended_preparation_policy(),
    )

    assert source.to_v2_bytes() == before
    assert len(prepared.constraints) == size * len(expected_kinds)
    assert len(prepared.one_hot_constraints) == size
    assert prepared.indicator_constraints == {}
    assert prepared.sos1_constraints == {}
    assert prepared.active_special_constraint_kinds == {SpecialConstraintKind.OneHot}
    assert {
        constraint.provenance[-1].kind for constraint in prepared.constraints.values()
    } == expected_kinds
    assert input_class.contains(prepared)
    assert OMMXFixstarsAmplifyAdapter(prepared).instance is prepared


@pytest.mark.parametrize(
    "special_constraints",
    ["none", "indicator", "sos1", "indicator-sos1"],
)
def test_one_hot_preparation_cases_have_the_same_feasible_states(
    special_constraints,
):
    size = 2
    instance = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
    )

    for choices in itertools.product(range(size), repeat=size):
        entries = {
            group * size + choice: float(choice == choices[group])
            for group in range(size)
            for choice in range(size)
        }
        assert instance.evaluate(State(entries=entries)).feasible
