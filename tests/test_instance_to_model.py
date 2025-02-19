import amplify
import pytest
from ommx.v1 import (
    Instance,
    Constraint,
    DecisionVariable,
    Linear,
    Quadratic,
    Polynomial,
)

from ommx_fixstars_amplify_adapter.exception import OMMXFixstarsAmplifyAdapterError
from ommx_fixstars_amplify_adapter.adapter import OMMXFixstarsAmplifyAdapter
from conftest import assert_amplify_model


def test_instance_to_model():
    """
    The function that converts from ommx.v1.Instance to amplify.Model.

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
    # Definition of Decision Variables (ommx.v1.DecisionVariable)
    decision_variables = [
        DecisionVariable.of_type(
            kind=DecisionVariable.BINARY, id=0, lower=0, upper=1, name="x"
        ),
        DecisionVariable.of_type(
            kind=DecisionVariable.INTEGER,
            id=1,
            lower=-20.0,
            upper=20.0,
            name="y",
            subscripts=[],
        ),
        DecisionVariable.of_type(
            kind=DecisionVariable.CONTINUOUS,
            id=2,
            lower=-30,
            upper=30,
            name="z",
            subscripts=[0],
        ),
        DecisionVariable.of_type(
            kind=DecisionVariable.CONTINUOUS,
            id=3,
            lower=float("-inf"),
            upper=float("inf"),
            name="w",
            subscripts=[1, 2],
        ),
    ]

    # Objective Function Definition: 2xyz + 3yz + 4z + 5
    objective = Polynomial(terms={(0, 1, 2): 2.0, (1, 2): 3.0, (2,): 4.0, (): 5.0})

    # Definition of Constraints
    constraints = []

    # constraint1: 6x + 7y + 8z - 9 <= 0
    constraint1_func = Linear(terms={0: 6.0, 1: 7.0, 2: 8.0}, constant=-9.0)
    constraint1 = Constraint(
        function=constraint1_func,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintA",
    )
    constraints.append(constraint1)

    # constraint2: 10xy + 11yz + 12xz -13 = 0
    constraint2_func = Quadratic(
        columns=[0, 1, 0],
        rows=[1, 2, 2],
        values=[10.0, 11.0, 12.0],
        linear=Linear(terms={}, constant=-13.0),
    )
    constraint2 = Constraint(
        function=constraint2_func, equality=Constraint.EQUAL_TO_ZERO, name="constraintB"
    )
    constraints.append(constraint2)

    # constraint3: 14xyz -15 >= 0
    constraint3_func = Polynomial(terms={(0, 1, 2): 14.0, (): -15.0})
    constraint3 = Constraint(
        function=constraint3_func * -1,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintC",
    )
    constraints.append(constraint3)

    # constraint4 :  w >= 16
    constraint4_func = Linear(terms={3: 1.0}, constant=-16.0)
    constraint4 = Constraint(
        function=constraint4_func * -1,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintD",
    )
    constraints.append(constraint4)

    # constraint5: w - 17 <= 0  (w <= 17)
    constraint5_func = Linear(terms={3: 1.0}, constant=-17.0)
    constraint5 = Constraint(
        function=constraint5_func,
        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        name="constraintE",
    )
    constraints.append(constraint5)

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


def test_error_unsupported_variable_kind():
    # Create OMMX instances with unsupported variable types
    decision_variables = [
        DecisionVariable.of_type(
            kind=DecisionVariable.SEMI_INTEGER, id=0, lower=0, upper=1, name="x"
        )
    ]

    instance = Instance.from_components(
        decision_variables=decision_variables,
        objective=0,
        constraints=[],
        sense=Instance.MINIMIZE,
    )

    with pytest.raises(OMMXFixstarsAmplifyAdapterError):
        OMMXFixstarsAmplifyAdapter(instance)
