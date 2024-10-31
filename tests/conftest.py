import typing

import amplify


def assert_amplify_model(model1: amplify.Model, model2: amplify.Model) -> None:
    """
    The Amplify.Model equivalence check function.
    """
    assert len(model1.variables) == len(model2.variables)
    for i in range(len(model1.variables)):
        assert model1.variables[i].id == model2.variables[i].id
        assert model1.variables[i].type == model2.variables[i].type
        assert model1.variables[i].lower_bound == model2.variables[i].lower_bound
        assert model1.variables[i].upper_bound == model2.variables[i].upper_bound

    def _is_equal_poly(
        expression1: typing.Union[amplify.Poly, amplify.Matrix],
        expression2: typing.Union[amplify.Poly, amplify.Matrix],
    ) -> None:
        if isinstance(expression1, amplify.Poly) and isinstance(
            expression2, amplify.Poly
        ):
            assert expression1.as_dict() == expression2.as_dict()
        elif isinstance(expression1, amplify.Matrix) and isinstance(
            expression2, amplify.Matrix
        ):
            assert expression1.to_poly().as_dict() == expression2.to_poly().as_dict()
        else:
            raise AssertionError()

    _is_equal_poly(model1.objective, model2.objective)

    assert len(model1.constraints) == len(model2.constraints)
    for i in range(len(model1.constraints)):
        _is_equal_poly(
            model1.constraints[i].conditional[0],
            model2.constraints[i].conditional[0],
        )
        assert (
            model1.constraints[i].conditional[1] == model2.constraints[i].conditional[1]
        )
        assert (
            model1.constraints[i].conditional[2] == model2.constraints[i].conditional[2]
        )
        assert model1.constraints[i].label == model2.constraints[i].label
