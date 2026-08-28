"""Tests that the one-hot (hints) formulation converts to the regular model.

The v2 Adapter ignores ``ConstraintHints``, so benchmarks measure only the
one-hot formulation and its rows represent the regular formulation as well.
This test pins the model identity that replaces the dropped regular runs.
"""

import pytest
from conftest import assert_amplify_model

from benchmarks.instance import build_assignment_instance, build_tsp_instance
from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter


@pytest.mark.parametrize(
    "build_instance",
    [build_assignment_instance, build_tsp_instance],
    ids=["assignment", "tsp"],
)
def test_one_hot_formulation_converts_to_the_regular_model(build_instance):
    size = 4
    regular = build_instance(size, formulation="regular")
    one_hot = build_instance(size, formulation="one-hot")

    regular_hints = regular.constraint_hints
    assert regular_hints is None or len(regular_hints.one_hot_constraints) == 0
    one_hot_hints = one_hot.constraint_hints
    assert one_hot_hints is not None
    assert len(one_hot_hints.one_hot_constraints) == 2 * size

    regular_model = OMMXFixstarsAmplifyAdapter(regular).solver_input
    one_hot_model = OMMXFixstarsAmplifyAdapter(one_hot).solver_input
    assert_amplify_model(regular_model, one_hot_model)
