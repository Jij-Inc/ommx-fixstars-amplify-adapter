"""Tests for the dedicated preparation benchmark workload."""

import copy
import itertools

import pytest
from conftest import assert_amplify_model
from ommx import ProvenanceKind, SpecialConstraintKind, State

from benchmarks.common import make_benchmark_operation
from benchmarks.instance import build_one_hot_preparation_instance
from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter


@pytest.mark.parametrize(
    ("special_constraints", "indicator_count", "sos1_count"),
    [
        ("indicator", 4, 0),
        ("sos1", 0, 4),
        ("indicator-sos1", 2, 2),
    ],
)
def test_direct_and_prepared_cases_have_aligned_active_constraints(
    special_constraints,
    indicator_count,
    sos1_count,
):
    size = 4
    direct = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
        preparation="none",
    )
    source = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
        preparation="recommended",
    )
    before = source.to_v2_bytes()
    input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS

    assert len(direct.constraints) == size
    assert len(direct.one_hot_constraints) == size
    assert direct.indicator_constraints == {}
    assert direct.sos1_constraints == {}
    assert len(source.constraints) == 0
    assert len(source.one_hot_constraints) == size
    assert len(source.indicator_constraints) == indicator_count
    assert len(source.sos1_constraints) == sos1_count
    assert not input_class.contains(source)

    prepared = copy.copy(source)
    prepared.prepare(
        input_class,
        OMMXFixstarsAmplifyAdapter.recommended_preparation_policy(),
    )

    assert source.to_v2_bytes() == before
    assert len(prepared.constraints) == size
    assert len(prepared.one_hot_constraints) == size
    assert prepared.indicator_constraints == {}
    assert prepared.sos1_constraints == {}
    assert len(prepared.removed_indicator_constraints) == indicator_count
    assert len(prepared.removed_sos1_constraints) == sos1_count
    assert prepared.active_special_constraint_kinds == {SpecialConstraintKind.OneHot}
    expected_kinds = (
        {ProvenanceKind.IndicatorConstraint, ProvenanceKind.Sos1Constraint}
        if indicator_count and sos1_count
        else {
            ProvenanceKind.IndicatorConstraint
            if indicator_count
            else ProvenanceKind.Sos1Constraint
        }
    )
    assert {
        constraint.provenance[-1].kind for constraint in prepared.constraints.values()
    } == expected_kinds
    assert input_class.contains(direct)
    assert input_class.contains(prepared)
    direct_model = OMMXFixstarsAmplifyAdapter(direct).solver_input
    prepared_model = OMMXFixstarsAmplifyAdapter(prepared).solver_input
    assert_amplify_model(direct_model, prepared_model)


@pytest.mark.parametrize(
    "special_constraints",
    ["indicator", "sos1", "indicator-sos1"],
)
def test_direct_source_and_prepared_cases_have_the_same_feasible_states(
    special_constraints,
):
    size = 2
    baseline = build_one_hot_preparation_instance(size)
    direct = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
        preparation="none",
    )
    source = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
        preparation="recommended",
    )
    input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS
    prepared = copy.copy(source)
    prepared.prepare(
        input_class,
        OMMXFixstarsAmplifyAdapter.recommended_preparation_policy(),
    )

    for values in itertools.product((0.0, 1.0), repeat=size**2):
        entries = dict(enumerate(values))
        expected = baseline.evaluate(State(entries=entries))
        for instance in (direct, source, prepared):
            evaluation = instance.evaluate(State(entries=entries))
            assert evaluation.feasible == expected.feasible
            assert evaluation.objective == expected.objective


@pytest.mark.parametrize(
    ("preparation", "method_name"),
    [
        ("none", "solve_without_preparation"),
        ("recommended", "solve"),
    ],
)
def test_end_to_end_uses_the_preparation_appropriate_api(
    monkeypatch,
    preparation,
    method_name,
):
    instance = build_one_hot_preparation_instance(
        2,
        special_constraints="indicator",
        preparation=preparation,
    )
    expected_solution = object()
    calls = []

    def fake_solve(cls, target, *, amplify_token, timeout):
        calls.append((cls, target, amplify_token, timeout))
        return expected_solution

    monkeypatch.setenv("AMPLIFY_TOKEN", "test-token")
    monkeypatch.setattr(
        OMMXFixstarsAmplifyAdapter,
        method_name,
        classmethod(fake_solve),
    )

    benchmark = make_benchmark_operation(
        "end-to-end",
        instance,
        solver_time_limit_ms=1234,
        special_constraints="indicator",
        preparation=preparation,
    )
    context = benchmark.setup()
    solution = benchmark.run(context)

    assert context is instance
    assert solution is expected_solution
    assert calls == [(OMMXFixstarsAmplifyAdapter, instance, "test-token", 1234)]
