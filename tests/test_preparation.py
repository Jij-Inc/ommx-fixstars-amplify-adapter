import copy

import pytest

from ommx import (
    ConstraintKind,
    DecisionVariable,
    Equality,
    IndicatorConstraint,
    IndicatorConstraintID,
    Instance,
    InstanceClassMismatch,
    OneHotConstraint,
    OneHotConstraintID,
    Sense,
    Sos1Constraint,
    Sos1ConstraintID,
)
from ommx.adapter import AdapterNotApplicableError
from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter


def test_recommended_preparation_policies_are_independent() -> None:
    first = OMMXFixstarsAmplifyAdapter.recommended_preparation_policy()
    second = OMMXFixstarsAmplifyAdapter.recommended_preparation_policy()

    assert first is not second
    first.special_constraints = None
    assert second.special_constraints is not None


def test_rejects_unsupported_special_constraints_without_mutating_input():
    indicator_variable = DecisionVariable.binary(0)
    continuous_variable = DecisionVariable.continuous(1, lower=-2, upper=2)

    instance = Instance.from_components(
        decision_variables=[indicator_variable, continuous_variable],
        objective=continuous_variable,
        active_constraints={},
        active_indicator_constraints={
            0: IndicatorConstraint(
                indicator_variable=indicator_variable,
                function=continuous_variable - 1,
                equality=Equality.LessThanOrEqualToZero,
            )
        },
        active_sos1_constraints={1: Sos1Constraint(variables=[continuous_variable])},
        sense=Sense.Minimize,
    )
    before = instance.to_v2_bytes()

    with pytest.raises(AdapterNotApplicableError) as error:
        OMMXFixstarsAmplifyAdapter(instance)
    mismatches = error.value.report.clause_reports[0].mismatches
    mismatch_types = {type(mismatch) for mismatch in mismatches}
    assert InstanceClassMismatch.IndicatorConstraintsNotAllowed in mismatch_types
    assert InstanceClassMismatch.Sos1ConstraintsNotAllowed in mismatch_types
    assert instance.to_v2_bytes() == before

    with pytest.raises(AdapterNotApplicableError) as error:
        OMMXFixstarsAmplifyAdapter.solve_without_preparation(
            instance,
            amplify_token="dummy",
        )
    mismatches = error.value.report.clause_reports[0].mismatches
    mismatch_types = {type(mismatch) for mismatch in mismatches}
    assert InstanceClassMismatch.IndicatorConstraintsNotAllowed in mismatch_types
    assert InstanceClassMismatch.Sos1ConstraintsNotAllowed in mismatch_types
    assert instance.to_v2_bytes() == before


def test_recommended_preparation_lowers_only_indicator_and_sos1() -> None:
    indicator = DecisionVariable.binary(0)
    one_hot_variables = [DecisionVariable.binary(i) for i in range(1, 3)]
    value = DecisionVariable.continuous(3, lower=0, upper=2)
    instance = Instance.from_components(
        decision_variables=[indicator, *one_hot_variables, value],
        objective=value,
        active_constraints={},
        active_indicator_constraints={30: (value <= 1).with_indicator(indicator)},
        active_one_hot_constraints={
            10: OneHotConstraint(variables=one_hot_variables),
        },
        active_sos1_constraints={20: Sos1Constraint(variables=one_hot_variables)},
        sense=Sense.Maximize,
    )
    before = instance.to_v2_bytes()
    input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS

    assert not OMMXFixstarsAmplifyAdapter.check_applicability(instance).is_member
    with pytest.raises(AdapterNotApplicableError):
        OMMXFixstarsAmplifyAdapter(instance)
    assert instance.to_v2_bytes() == before

    prepared = copy.copy(instance)
    prepared.prepare(
        input_class,
        OMMXFixstarsAmplifyAdapter.recommended_preparation_policy(),
    )

    assert IndicatorConstraintID(30) in instance.active_constraints
    assert Sos1ConstraintID(20) in instance.active_constraints
    assert not prepared.active_constraints.of_kind(ConstraintKind.Indicator)
    assert set(prepared.active_constraints.of_kind(ConstraintKind.OneHot)) == {
        OneHotConstraintID(10)
    }
    assert not prepared.active_constraints.of_kind(ConstraintKind.Sos1)
    assert prepared.active_special_constraint_kinds == {
        ConstraintKind.OneHot,
    }
    assert input_class.contains(prepared)
    assert OMMXFixstarsAmplifyAdapter.check_applicability(prepared).is_member

    adapter = OMMXFixstarsAmplifyAdapter(prepared)
    assert adapter.instance is prepared
