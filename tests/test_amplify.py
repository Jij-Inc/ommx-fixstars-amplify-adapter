import amplify


def test_model():
    """
    The test to ensure that the methods of amplify.Model work as intended.

    Minimize: 2xyz + 3yz + 4z + 5
    Subject to:
        6x + 7y + 8z <= 9
        10xy + 11yz + 12xz = 13
        14xyz >= 15
        16 <= w <= 17
        x: Binary
        y: Integer (lower bound: -20, upper bound: 20)
        z: Continuous (lower bound: -30, upper bound: 30)
        z: Continuous (lower bound: -inf, upper bound: inf)
    """
    gen = amplify.VariableGenerator()
    x = gen.scalar("Binary")
    y = gen.scalar("Integer", bounds=(-20, 20))
    z = gen.scalar("Real", bounds=(-30, 30))
    w = gen.scalar("Real")
    model = amplify.Model()
    model += 2.0 * x * y * z + 3.0 * y * z + 4.0 * z + 5.0
    model += amplify.less_equal(6.0 * x + 7.0 * y + 8.0 * z, 9.0)
    model += amplify.equal_to(10.0 * x * y + 11.0 * y * z + 12.0 * x * z, 13.0)
    model += amplify.greater_equal(14.0 * x * y * z, 15.0)
    model += amplify.clamp(w, (16, 17))

    # Check decision variables
    assert len(model.variables) == 4
    assert model.variables[0].id == 0
    assert model.variables[0].type == amplify.VariableType.Binary
    assert model.variables[1].id == 1
    assert model.variables[1].type == amplify.VariableType.Integer
    assert model.variables[1].lower_bound == -20
    assert model.variables[1].upper_bound == 20
    assert model.variables[2].id == 2
    assert model.variables[2].type == amplify.VariableType.Real
    assert model.variables[2].lower_bound == -30
    assert model.variables[2].upper_bound == 30
    assert model.variables[3].id == 3
    assert model.variables[3].type == amplify.VariableType.Real
    assert model.variables[3].lower_bound is None
    assert model.variables[3].upper_bound is None
    # Check the objective function
    assert isinstance(model.objective, amplify.Poly)
    assert model.objective.as_dict() == {
        (0, 1, 2): 2.0,
        (1, 2): 3.0,
        (2,): 4.0,
        (): 5.0,
    }
    # Check the number of constraints
    assert len(model.constraints) == 4
    for constraint in model.constraints:
        assert len(constraint.conditional) == 3
    # Check the first constraint
    assert model.constraints[0].conditional[0].as_dict() == {
        (0,): 6.0,
        (1,): 7.0,
        (2,): 8.0,
    }
    assert model.constraints[0].conditional[1] == "LE"
    assert model.constraints[0].conditional[2] == 9.0
    # Check the second constraint
    assert model.constraints[1].conditional[0].as_dict() == {
        (0, 1): 10.0,
        (1, 2): 11.0,
        (0, 2): 12.0,
    }
    assert model.constraints[1].conditional[1] == "EQ"
    assert model.constraints[1].conditional[2] == 13.0
    # Check the third constraint
    assert model.constraints[2].conditional[0].as_dict() == {(0, 1, 2): 14.0}
    assert model.constraints[2].conditional[1] == "GE"
    assert model.constraints[2].conditional[2] == 15.0
    # Check the fourth constraint
    assert model.constraints[3].conditional[0].as_dict() == {(3,): 1.0}
    assert model.constraints[3].conditional[1] == "BW"
    assert model.constraints[3].conditional[2] == (16, 17)


def test_poly_degree():
    poly0 = amplify.Poly()
    assert poly0.degree() == -1

    objective = amplify.Model().objective
    assert isinstance(objective, amplify.Poly)
    assert objective.degree() == -1

    gen = amplify.VariableGenerator()
    x = gen.array("Integer", 4)

    assert (x[0]).degree() == 1
    assert (x[0] * x[1]).degree() == 2
    assert (x[0] * x[1] * x[2]).degree() == 3
    assert (x[0] * x[1] * x[2] * x[3]).degree() == 4

    assert (x[0] + x[1]).degree() == 1
    assert (x[0] * x[1] + x[2]).degree() == 2
    assert (x[0] * x[1] + x[2] * x[3]).degree() == 2
    assert (x[0] * x[1] * x[2] + x[3]).degree() == 3
