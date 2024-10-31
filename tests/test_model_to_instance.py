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

    assert len(ommx_instance.raw.decision_variables) == 4
    # Check the decision variable `x`
    assert ommx_instance.raw.decision_variables[0].id == 0
    assert ommx_instance.raw.decision_variables[0].kind == DecisionVariable.BINARY
    assert ommx_instance.raw.decision_variables[0].name == "x"
    assert ommx_instance.raw.decision_variables[0].bound.lower == 0
    assert ommx_instance.raw.decision_variables[0].bound.upper == 1
    # Check the decision variable `y`
    assert ommx_instance.raw.decision_variables[1].id == 1
    assert ommx_instance.raw.decision_variables[1].kind == DecisionVariable.INTEGER
    assert ommx_instance.raw.decision_variables[1].name == "y"
    assert ommx_instance.raw.decision_variables[1].bound.lower == -20
    assert ommx_instance.raw.decision_variables[1].bound.upper == 20
    # Check the decision variable `z`
    assert ommx_instance.raw.decision_variables[2].id == 2
    assert ommx_instance.raw.decision_variables[2].kind == DecisionVariable.CONTINUOUS
    assert ommx_instance.raw.decision_variables[2].name == "z"
    assert ommx_instance.raw.decision_variables[2].bound.lower == -30
    assert ommx_instance.raw.decision_variables[2].bound.upper == 30
    # Check the decision variable `w`
    assert ommx_instance.raw.decision_variables[3].id == 3
    assert ommx_instance.raw.decision_variables[3].kind == DecisionVariable.CONTINUOUS
    assert ommx_instance.raw.decision_variables[3].name == "w"
    assert ommx_instance.raw.decision_variables[3].bound.lower == float("-inf")
    assert ommx_instance.raw.decision_variables[3].bound.upper == float("inf")

    # Check the objective function: 2xyz + 3yz + 4z + 5
    objective = ommx_instance.raw.objective
    assert objective.HasField("polynomial")
    assert len(objective.polynomial.terms) == 4
    assert objective.polynomial.terms[0].ids == [0, 1, 2]
    assert objective.polynomial.terms[0].coefficient == 2.0
    assert objective.polynomial.terms[1].ids == [1, 2]
    assert objective.polynomial.terms[1].coefficient == 3.0
    assert objective.polynomial.terms[2].ids == [2]
    assert objective.polynomial.terms[2].coefficient == 4.0
    assert objective.polynomial.terms[3].ids == []
    assert objective.polynomial.terms[3].coefficient == 5.0

    # Check the number of constraints
    assert len(ommx_instance.raw.constraints) == 5

    # Check the first constraint: 6x + 7y + 8z -9 <= 0
    constraint1 = ommx_instance.raw.constraints[0]
    assert constraint1.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    assert constraint1.function.HasField("linear")
    assert len(constraint1.function.linear.terms) == 3
    assert constraint1.function.linear.terms[0].id == 0
    assert constraint1.function.linear.terms[0].coefficient == 6.0
    assert constraint1.function.linear.terms[1].id == 1
    assert constraint1.function.linear.terms[1].coefficient == 7.0
    assert constraint1.function.linear.terms[2].id == 2
    assert constraint1.function.linear.terms[2].coefficient == 8.0
    assert constraint1.function.linear.constant == -9.0

    # Check the second constraint: 10xy + 11yz + 12xz -13 = 0
    constraint2 = ommx_instance.raw.constraints[1]
    assert constraint2.equality == Constraint.EQUAL_TO_ZERO
    assert constraint2.function.HasField("quadratic")
    assert len(constraint2.function.quadratic.columns) == 3
    assert len(constraint2.function.quadratic.rows) == 3
    assert len(constraint2.function.quadratic.values) == 3
    assert constraint2.function.quadratic.columns[0] == 0
    assert constraint2.function.quadratic.rows[0] == 1
    assert constraint2.function.quadratic.values[0] == 10
    assert constraint2.function.quadratic.columns[1] == 1
    assert constraint2.function.quadratic.rows[1] == 2
    assert constraint2.function.quadratic.values[1] == 11
    assert constraint2.function.quadratic.columns[2] == 0
    assert constraint2.function.quadratic.rows[2] == 2
    assert constraint2.function.quadratic.values[2] == 12
    assert constraint2.function.quadratic.linear.terms == []
    assert constraint2.function.quadratic.linear.constant == -13.0

    # Check the third constraint: 14xyz -15 <= 0
    constraint3 = ommx_instance.raw.constraints[2]
    assert constraint3.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    assert constraint3.function.HasField("polynomial")
    assert len(constraint3.function.polynomial.terms) == 2
    assert constraint3.function.polynomial.terms[0].ids == [0, 1, 2]
    assert constraint3.function.polynomial.terms[0].coefficient == -14.0
    assert constraint3.function.polynomial.terms[1].ids == []
    assert constraint3.function.polynomial.terms[1].coefficient == 15.0

    # Check the fourth constraint: 16 <= w <= 17
    constraint4 = ommx_instance.raw.constraints[3]
    assert constraint4.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    assert constraint4.function.HasField("linear")
    assert len(constraint4.function.linear.terms) == 1
    assert constraint4.function.linear.terms[0].id == 3
    assert constraint4.function.linear.terms[0].coefficient == -1.0
    assert constraint4.function.linear.constant == 16.0
    constraint5 = ommx_instance.raw.constraints[4]
    assert constraint5.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
    assert constraint5.function.HasField("linear")
    assert len(constraint5.function.linear.terms) == 1
    assert constraint5.function.linear.terms[0].id == 3
    assert constraint5.function.linear.terms[0].coefficient == 1.0
    assert constraint5.function.linear.constant == -17.0


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
    assert decision_variable[0].raw.id == 0
    assert decision_variable[0].raw.kind == DecisionVariable.BINARY
    assert decision_variable[0].raw.bound.lower == 0
    assert decision_variable[0].raw.bound.upper == 1
    assert decision_variable[1].raw.id == 1
    assert decision_variable[1].raw.kind == DecisionVariable.INTEGER
    assert decision_variable[1].raw.bound.lower == float("-inf")
    assert decision_variable[1].raw.bound.upper == float("inf")
    assert decision_variable[1].raw.name == "y"
    assert decision_variable[2].raw.id == 2
    assert decision_variable[2].raw.kind == DecisionVariable.CONTINUOUS
    assert decision_variable[2].raw.bound.lower == -30
    assert decision_variable[2].raw.bound.upper == 30


def test_error_ising_variable():
    gen = amplify.VariableGenerator()
    x = gen.scalar("Ising")
    model = amplify.Model(x)

    with pytest.raises(OMMXFixstarsAmplifyAdapterError):
        model_to_instance(model)
