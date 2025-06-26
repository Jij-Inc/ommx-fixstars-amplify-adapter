import amplify
import pytest
from ommx.v1 import Constraint, DecisionVariable

from ommx_fixstars_amplify_adapter.amplify_to_ommx import (
    model_to_instance,
    OMMXInstanceBuilder,
)
from ommx_fixstars_amplify_adapter.exception import OMMXFixstarsAmplifyAdapterError


def test_model_to_instance():
    """
    The function that converts from amplify.Model to ommx.v1.Instance.

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
    gen = amplify.VariableGenerator()
    x = gen.scalar("Binary", name="x")
    y = gen.scalar("Integer", name="y", bounds=(-20, 20))
    z = gen.scalar("Real", name="z", bounds=(-30, 30))
    w = gen.scalar("Real", name="w")
    model = amplify.Model()
    model += 2.0 * x * y * z + 3.0 * y * z + 4.0 * z + 5.0
    model += amplify.less_equal(6.0 * x + 7.0 * y + 8.0 * z, 9.0)
    model += amplify.equal_to(10.0 * x * y + 11.0 * y * z + 12.0 * x * z, 13.0)
    model += amplify.greater_equal(14.0 * x * y * z, 15.0)
    model += amplify.clamp(w, (16, 17))
    ommx_instance = model_to_instance(model)

    assert len(ommx_instance.decision_variables) == 4
    # Check the decision variable `x`
    assert ommx_instance.get_decision_variable_by_id(0).kind == DecisionVariable.BINARY
    assert ommx_instance.get_decision_variable_by_id(0).name == "x"
    assert ommx_instance.get_decision_variable_by_id(0).bound.lower == 0
    assert ommx_instance.get_decision_variable_by_id(0).bound.upper == 1
    # Check the decision variable `y`
    assert ommx_instance.get_decision_variable_by_id(1).kind == DecisionVariable.INTEGER
    assert ommx_instance.get_decision_variable_by_id(1).name == "y"
    assert ommx_instance.get_decision_variable_by_id(1).bound.lower == -20
    assert ommx_instance.get_decision_variable_by_id(1).bound.upper == 20
    # Check the decision variable `z`
    assert (
        ommx_instance.get_decision_variable_by_id(2).kind == DecisionVariable.CONTINUOUS
    )
    assert ommx_instance.get_decision_variable_by_id(2).name == "z"
    assert ommx_instance.get_decision_variable_by_id(2).bound.lower == -30
    assert ommx_instance.get_decision_variable_by_id(2).bound.upper == 30
    # Check the decision variable `w`
    assert (
        ommx_instance.get_decision_variable_by_id(3).kind == DecisionVariable.CONTINUOUS
    )
    assert ommx_instance.get_decision_variable_by_id(3).name == "w"
    assert ommx_instance.get_decision_variable_by_id(3).bound.lower == float("-inf")
    assert ommx_instance.get_decision_variable_by_id(3).bound.upper == float("inf")

    # Check the objective function: 2xyz + 3yz + 4z + 5
    assert ommx_instance.objective.terms == {
        (0, 1, 2): 2.0,
        (1, 2): 3.0,
        (2,): 4.0,
        (): 5.0,
    }

    # Check the number of constraints
    assert len(ommx_instance.constraints) == 5

    # Check the first constraint: 6x + 7y + 8z -9 <= 0
    assert (
        ommx_instance.get_constraint_by_id(0).equality
        == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    )
    assert ommx_instance.get_constraint_by_id(0).function.terms == {
        (0,): 6.0,
        (1,): 7.0,
        (2,): 8.0,
        (): -9.0,
    }

    # Check the second constraint: 10xy + 11yz + 12xz -13 = 0
    assert ommx_instance.get_constraint_by_id(1).equality == Constraint.EQUAL_TO_ZERO
    assert ommx_instance.get_constraint_by_id(1).function.terms == {
        (0, 1): 10.0,
        (1, 2): 11.0,
        (0, 2): 12.0,
        (): -13.0,
    }

    # Check the third constraint: 14xyz -15 >= 0
    assert (
        ommx_instance.get_constraint_by_id(2).equality
        == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    )
    assert ommx_instance.get_constraint_by_id(2).function.terms == {
        (0, 1, 2): -14.0,
        (): 15.0,
    }

    # Check the fourth constraint: 16 <= w <= 17
    assert (
        ommx_instance.get_constraint_by_id(3).equality
        == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    )
    assert ommx_instance.get_constraint_by_id(3).function.terms == {
        (3,): -1.0,
        (): 16.0,
    }
    assert (
        ommx_instance.get_constraint_by_id(4).equality
        == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    )
    assert ommx_instance.get_constraint_by_id(4).function.terms == {
        (3,): 1.0,
        (): -17.0,
    }


def test_builder_decision_variable():
    gen = amplify.VariableGenerator()
    model = amplify.Model()
    x = gen.scalar("Binary")
    y = gen.scalar("Integer", name="y")
    z = gen.scalar("Real", bounds=(-30, 30))
    model += x + y + z

    builder = OMMXInstanceBuilder(model)
    decision_variable = builder.decision_variables()

    assert len(decision_variable) == 3
    assert decision_variable[0].id == 0
    assert decision_variable[0].kind == DecisionVariable.BINARY
    assert decision_variable[0].bound.lower == 0
    assert decision_variable[0].bound.upper == 1
    assert decision_variable[1].id == 1
    assert decision_variable[1].kind == DecisionVariable.INTEGER
    assert decision_variable[1].bound.lower == float("-inf")
    assert decision_variable[1].bound.upper == float("inf")
    assert decision_variable[1].name == "y"
    assert decision_variable[2].id == 2
    assert decision_variable[2].kind == DecisionVariable.CONTINUOUS
    assert decision_variable[2].bound.lower == -30
    assert decision_variable[2].bound.upper == 30


def test_error_ising_variable():
    gen = amplify.VariableGenerator()
    x = gen.scalar("Ising")
    model = amplify.Model(x)

    with pytest.raises(OMMXFixstarsAmplifyAdapterError):
        model_to_instance(model)
