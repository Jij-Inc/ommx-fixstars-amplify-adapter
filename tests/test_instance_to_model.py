import amplify
import pytest
from ommx.adapter import AdapterNotApplicableError
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
    Polynomial,
    Quadratic,
    Sense,
)

from ommx_fixstars_amplify_adapter.adapter import OMMXFixstarsAmplifyAdapter
from ommx_fixstars_amplify_adapter.exception import OMMXFixstarsAmplifyAdapterError
from conftest import assert_amplify_model


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


def test_instance_to_model():
    """
    The function that converts from ommx.Instance to amplify.Model.

    Minimize: 2xyz + 3yz + 4z + 5
    Subject to:
        6x + 7y + 8z <= 9
        10xy + 11yz + 12xz = 13
        14xyz >= 15
        16 <= w <= 17
        x: Binary
        y: Integer (lower bound: -20, upper bound: 20)
        z: Continuous (lower bound: -30, upper bound: 30)
        w: Continuous (lower bound: -inf, upper bound: inf)
    """
    # Definition of Decision Variables (ommx.DecisionVariable)
    decision_variables = [
        DecisionVariable.binary(id=0, name="x"),
        DecisionVariable.integer(id=1, lower=-20, upper=20, name="y"),
        DecisionVariable.continuous(
            id=2, lower=-30, upper=30, name="z", subscripts=[0]
        ),
        DecisionVariable.continuous(
            id=3, lower=float("-inf"), upper=float("inf"), name="w", subscripts=[1, 2]
        ),
    ]

    # Objective Function Definition: 2xyz + 3yz + 4z + 5
    objective = Polynomial(terms={(0, 1, 2): 2.0, (1, 2): 3.0, (2,): 4.0, (): 5.0})

    # Definition of Constraints
    constraints = {}

    # constraint0: 6x + 7y + 8z - 9 <= 0
    constraint0_func = Linear(terms={0: 6.0, 1: 7.0, 2: 8.0}, constant=-9.0)
    constraint0 = Constraint(
        function=constraint0_func,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintA",
    )
    constraints[0] = constraint0

    # constraint1: 10xy + 11yz + 12xz -13 = 0
    constraint1_func = Quadratic(
        columns=[0, 1, 0],
        rows=[1, 2, 2],
        values=[10.0, 11.0, 12.0],
        linear=Linear(terms={}, constant=-13.0),
    )
    constraint1 = Constraint(
        function=constraint1_func, equality=Constraint.EQUAL_TO_ZERO, name="constraintB"
    )
    constraints[1] = constraint1

    # constraint2: 14xyz -15 >= 0
    constraint2_func = Polynomial(terms={(0, 1, 2): 14.0, (): -15.0})
    constraint2 = Constraint(
        function=constraint2_func * -1,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintC",
    )
    constraints[2] = constraint2

    # constraint3 :  w >= 16
    constraint3_func = Linear(terms={3: 1.0}, constant=-16.0)
    constraint3 = Constraint(
        function=constraint3_func * -1,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintD",
    )
    constraints[3] = constraint3

    # constraint4: w - 17 <= 0  (w <= 17)
    constraint4_func = Linear(terms={3: 1.0}, constant=-17.0)
    constraint4 = Constraint(
        function=constraint4_func,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintE",
    )
    constraints[4] = constraint4

    # Creating an OMMX instance
    instance = Instance.from_components(
        decision_variables=decision_variables,
        objective=objective,
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    model = adapter.solver_input

    # Construct the expected model
    gen = amplify.VariableGenerator()
    x = gen.scalar("Binary", name="x")
    y = gen.scalar("Integer", bounds=(-20, 20), name="y")
    z = gen.scalar("Real", bounds=(-30, 30), name="z_{0}")
    w = gen.scalar("Real", bounds=(float("-inf"), float("inf")), name="w_{1, 2}")

    expected_model = amplify.Model()
    expected_model += 2.0 * (x * y * z) + 3.0 * (y * z) + 4.0 * z + 5.0
    expected_model += amplify.less_equal(
        6 * x + 7 * y + 8 * z - 9, 0, label="constraintA [id: 0]"
    )
    expected_model += amplify.equal_to(
        10.0 * x * y + 11.0 * y * z + 12.0 * x * z - 13.0,
        0,
        label="constraintB [id: 1]",
    )
    expected_model += amplify.less_equal(
        -14.0 * x * y * z + 15.0, 0.0, label="constraintC [id: 2]"
    )
    expected_model += amplify.less_equal(-1 * w + 16, 0, label="constraintD [id: 3]")
    expected_model += amplify.less_equal(w - 17, 0, label="constraintE [id: 4]")

    assert_amplify_model(model, expected_model)


@pytest.mark.parametrize(
    ("equality", "constant"),
    [
        (Equality.EqualToZero, 0.0),
        (Equality.LessThanOrEqualToZero, -1.0),
    ],
)
def test_skips_feasible_constant_constraint(
    equality: Equality, constant: float
) -> None:
    instance = Instance.from_components(
        decision_variables=[],
        objective=0,
        constraints={
            0: Constraint(
                function=constant,
                equality=equality,
            )
        },
        sense=Sense.Minimize,
    )

    adapter = OMMXFixstarsAmplifyAdapter(instance)

    assert len(adapter.solver_input.constraints) == 0


@pytest.mark.parametrize(
    ("equality", "constant"),
    [
        (Equality.EqualToZero, 1.0),
        (Equality.LessThanOrEqualToZero, 1.0),
    ],
)
def test_rejects_infeasible_constant_constraint(
    equality: Equality, constant: float
) -> None:
    instance = Instance.from_components(
        decision_variables=[],
        objective=0,
        constraints={
            0: Constraint(
                function=constant,
                equality=equality,
            )
        },
        sense=Sense.Minimize,
    )

    with pytest.raises(
        OMMXFixstarsAmplifyAdapterError,
        match="Infeasible constant constraint was found: id 0",
    ):
        OMMXFixstarsAmplifyAdapter(instance)


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


def test_one_hot_constraint():
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]

    instance = Instance.from_components(
        decision_variables=x,
        objective=0,
        constraints={},
        one_hot_constraints={
            0: OneHotConstraint(variables=[x[0], x[1], x[2]], name="one_hot_constraint")
        },
        sense=Instance.MINIMIZE,
    )

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    model = adapter.solver_input

    # Construct the expected model
    gen = amplify.VariableGenerator()
    y0 = gen.scalar("Binary", name="x_{0}")
    y1 = gen.scalar("Binary", name="x_{1}")
    y2 = gen.scalar("Binary", name="x_{2}")

    expected_model = amplify.Model()
    expected_model += amplify.one_hot(y0 + y1 + y2, label="one_hot_constraint [id: 0]")

    assert_amplify_model(model, expected_model)


def test_regular_and_one_hot_constraints():
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]

    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] + 2 * x[1],
        constraints={0: 3 * x[0] + 5 * x[2] <= 1},
        one_hot_constraints={
            0: OneHotConstraint(variables=[x[0], x[1], x[2]], name="one_hot_constraint")
        },
        sense=Instance.MINIMIZE,
    )

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    model = adapter.solver_input

    # Construct the expected model
    gen = amplify.VariableGenerator()
    y0 = gen.scalar("Binary", name="x_{0}")
    y1 = gen.scalar("Binary", name="x_{1}")
    y2 = gen.scalar("Binary", name="x_{2}")

    expected_model = amplify.Model()
    expected_model += y0 + 2 * y1
    expected_model += amplify.one_hot(y0 + y1 + y2, label="one_hot_constraint [id: 0]")
    expected_model += amplify.less_equal(3 * y0 + 5 * y2 - 1, 0, label="None [id: 0]")

    assert_amplify_model(model, expected_model)


def test_multiple_one_hot_constraints():
    # 2x2 assignment: each variable belongs to one row and one column one-hot
    x = [
        DecisionVariable.binary(i, name="x", subscripts=[i // 2, i % 2])
        for i in range(4)
    ]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] + 2 * x[3],
        constraints={},
        one_hot_constraints={
            0: OneHotConstraint(variables=[x[0], x[1]], name="row"),
            1: OneHotConstraint(variables=[x[2], x[3]], name="row"),
            2: OneHotConstraint(variables=[x[0], x[2]], name="col"),
            3: OneHotConstraint(variables=[x[1], x[3]], name="col"),
        },
        sense=Instance.MINIMIZE,
    )

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    model = adapter.solver_input

    # Construct the expected model
    gen = amplify.VariableGenerator()
    y = [gen.scalar("Binary", name=f"x_{{{i // 2}, {i % 2}}}") for i in range(4)]

    expected_model = amplify.Model()
    expected_model += y[0] + 2 * y[3]
    expected_model += amplify.one_hot(y[0] + y[1], label="row [id: 0]")
    expected_model += amplify.one_hot(y[2] + y[3], label="row [id: 1]")
    expected_model += amplify.one_hot(y[0] + y[2], label="col [id: 2]")
    expected_model += amplify.one_hot(y[1] + y[3], label="col [id: 3]")

    assert_amplify_model(model, expected_model)


def test_partial_evaluate():
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=1 * x[0] + 2 * x[1] + 3 * x[2],
        constraints={0: 1 * x[0] + 2 * x[1] + 3 * x[2] <= 2},
        sense=Instance.MINIMIZE,
    )
    assert instance.used_decision_variables == x
    partial = instance.partial_evaluate({0: 1})
    # x[0] is no longer present in the problem
    assert partial.used_decision_variables == x[1:]

    adapter = OMMXFixstarsAmplifyAdapter(partial)
    model = adapter.solver_input

    gen = amplify.VariableGenerator()
    y1 = gen.scalar("Binary", name="x_{1}")
    y2 = gen.scalar("Binary", name="x_{2}")

    expected_model = amplify.Model()
    expected_model += 2.0 * y1 + 3.0 * y2 + 1.0
    expected_model += amplify.less_equal(
        2.0 * y1 + 3.0 * y2 - 1.0, 0, label="None [id: 0]"
    )

    assert_amplify_model(model, expected_model)

    partial = instance.partial_evaluate({1: 1})
    assert partial.used_decision_variables == [x[0], x[2]]

    adapter = OMMXFixstarsAmplifyAdapter(partial)
    model = adapter.solver_input

    gen2 = amplify.VariableGenerator()
    y0 = gen2.scalar("Binary", name="x_{0}")
    y2 = gen2.scalar("Binary", name="x_{2}")

    expected_model_2 = amplify.Model()
    expected_model_2 += 1.0 * y0 + 3.0 * y2 + 2.0
    expected_model_2 += amplify.less_equal(
        1.0 * y0 + 3.0 * y2 - 0.0, 0, label="None [id: 0]"
    )

    assert_amplify_model(model, expected_model_2)

    partial = instance.partial_evaluate({2: 1})
    assert partial.used_decision_variables == x[0:2]

    adapter = OMMXFixstarsAmplifyAdapter(partial)
    model = adapter.solver_input

    gen3 = amplify.VariableGenerator()
    y0 = gen3.scalar("Binary", name="x_{0}")
    y1 = gen3.scalar("Binary", name="x_{1}")

    expected_model_3 = amplify.Model()
    expected_model_3 += 1.0 * y0 + 2.0 * y1 + 3.0
    expected_model_3 += amplify.less_equal(
        1.0 * y0 + 2.0 * y1 - (-1.0), 0, label="None [id: 0]"
    )

    assert_amplify_model(model, expected_model_3)


def test_relax_constraint():
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] + x[1],
        constraints={0: x[0] + 2 * x[1] <= 1, 1: x[1] + x[2] <= 1},
        sense=Instance.MINIMIZE,
    )

    assert instance.used_decision_variables == x
    instance.relax_constraint(1, "relax")
    # id for x[2] is listed as irrelevant
    assert instance.irrelevant_decision_variable_ids() == {x[2].id}

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    model = adapter.solver_input

    gen = amplify.VariableGenerator()
    y0 = gen.scalar("Binary", name="x_{0}")
    y1 = gen.scalar("Binary", name="x_{1}")

    expected_model = amplify.Model()
    expected_model += 1.0 * y0 + 1.0 * y1
    expected_model += amplify.less_equal(
        1.0 * y0 + 2.0 * y1 - 1.0, 0, label="None [id: 0]"
    )

    assert_amplify_model(model, expected_model)
