import pytest

from ommx import (
    Constraint,
    DecisionVariable,
    DegreeBound,
    Equality,
    Instance,
    InstanceClassMismatch,
    Kind,
    Linear,
    OneHotConstraint,
    Sense,
)
from ommx.adapter import AdapterNotApplicableError

from ommx_fixstars_amplify_adapter.adapter import OMMXFixstarsAmplifyAdapter


def test_declares_polynomial_input_class() -> None:
    input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS
    [clause] = input_class.clauses

    assert clause.label == "fixstars-amplify-polynomial"
    assert clause.allowed_variable_kinds == {
        Kind.Binary,
        Kind.Integer,
        Kind.Continuous,
    }
    assert clause.objective_degree_bound == DegreeBound.unbounded()
    assert clause.regular_constraint_degree_bounds == {
        Equality.EqualToZero: DegreeBound.unbounded(),
        Equality.LessThanOrEqualToZero: DegreeBound.unbounded(),
    }
    assert clause.indicator_constraint_degree_bounds == {}
    assert clause.allows_one_hot
    assert not clause.allows_sos1
    assert clause.allowed_senses == {Sense.Minimize, Sense.Maximize}


def test_input_class_accepts_polynomial_instance():
    binary = [DecisionVariable.binary(i) for i in range(2)]
    integer = DecisionVariable.integer(2, lower=-2, upper=2)
    continuous = DecisionVariable.continuous(3, lower=-2, upper=2)

    instance = Instance.from_components(
        decision_variables=[*binary, integer, continuous],
        objective=binary[0] + binary[1] + integer + continuous,
        constraints={0: integer + continuous == 1, 1: integer - continuous <= 2},
        one_hot_constraints={0: OneHotConstraint(variables=binary)},
        sense=Sense.Minimize,
    )
    before = instance.to_v2_bytes()

    report = OMMXFixstarsAmplifyAdapter.check_applicability(instance)
    assert report.is_member
    assert report.matching_clauses == [(0, "fixstars-amplify-polynomial")]

    OMMXFixstarsAmplifyAdapter(instance)
    assert instance.to_v2_bytes() == before


@pytest.mark.parametrize(
    ("decision_variable", "kind"),
    [
        ([DecisionVariable.semi_integer(0, lower=1, upper=3)], Kind.SemiInteger),
        (
            [DecisionVariable.semi_continuous(0, lower=1, upper=3)],
            Kind.SemiContinuous,
        ),
    ],
)
def test_rejects_unsupported_variable_kinds(decision_variable, kind):
    constraint = Constraint(
        function=Linear(terms={0: 1.0}, constant=-5.0),
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
    )

    instance = Instance.from_components(
        decision_variables=decision_variable,
        objective=Linear(terms={0: 1.0}),
        constraints={0: constraint},
        sense=Instance.MINIMIZE,
    )

    with pytest.raises(AdapterNotApplicableError) as error:
        OMMXFixstarsAmplifyAdapter(instance)

    mismatches = error.value.report.clause_reports[0].mismatches
    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert isinstance(mismatch, InstanceClassMismatch.VariableKindNotAllowed)
    assert mismatch.kind == kind
    assert mismatch.variable_ids == {0}


def test_accepts_unused_unsupported_variable_kind_without_mutating_input():
    used = DecisionVariable.binary(0)
    unused = DecisionVariable.semi_integer(1, lower=1, upper=3)
    instance = Instance.from_components(
        decision_variables=[used, unused],
        objective=2 * used + 1,
        constraints={},
        sense=Sense.Minimize,
    )
    before = instance.to_v2_bytes()

    report = OMMXFixstarsAmplifyAdapter.check_applicability(instance)
    adapter = OMMXFixstarsAmplifyAdapter(instance)

    assert report.is_member
    assert set(adapter.variable_map) == {used.id}
    assert instance.to_v2_bytes() == before
