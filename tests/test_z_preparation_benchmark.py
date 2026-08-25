"""Tests for the OMMX v2 preparation benchmark baseline."""

from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter

from benchmarks.instance import build_one_hot_preparation_instance


def test_one_hot_preparation_baseline():
    size = 3
    instance = build_one_hot_preparation_instance(size)

    assert len(instance.decision_variables) == size**2
    assert len(instance.constraints) == size
    assert instance.constraint_hints is not None
    assert len(instance.constraint_hints.one_hot_constraints) == size
    assert len(OMMXFixstarsAmplifyAdapter(instance).solver_input.variables) == size**2
