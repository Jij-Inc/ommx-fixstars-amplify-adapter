import typing
from dataclasses import dataclass

import amplify
from ommx.v1 import (
    Constraint,
    DecisionVariable,
    Instance,
    Linear,
    Polynomial,
    Quadratic,
    Function,
)

from .exception import OMMXFixstarsAmplifyAdapterError


@dataclass
class OMMXInstanceBuilder:
    """
    Build ommx.v1.Instance from the Model of Fixstars Amplify.
    """

    model: amplify.Model

    def decision_variables(self) -> typing.List[DecisionVariable]:
        decision_variables = []

        for var in self.model.variables:
            # TODO: How to deal with the case where the variable is an ising variable.
            if var.type == amplify.VariableType.Binary:
                kind = DecisionVariable.BINARY
                lower = 0
                upper = 1
            elif var.type == amplify.VariableType.Integer:
                kind = DecisionVariable.INTEGER
                lower = float("-inf") if var.lower_bound is None else var.lower_bound
                upper = float("inf") if var.upper_bound is None else var.upper_bound
            elif var.type == amplify.VariableType.Real:
                kind = DecisionVariable.CONTINUOUS
                lower = float("-inf") if var.lower_bound is None else var.lower_bound
                upper = float("inf") if var.upper_bound is None else var.upper_bound
            elif var.type == amplify.VariableType.Ising:
                raise OMMXFixstarsAmplifyAdapterError(
                    "Ising variable is not supported now. Please use the Binary variable."
                )
            else:
                raise OMMXFixstarsAmplifyAdapterError(
                    f"Unintended variable type: {var.type}"
                )

            decision_variables.append(
                DecisionVariable.of_type(
                    kind=kind,
                    id=var.id,
                    lower=lower,
                    upper=upper,
                    name=var.name,
                )
            )

        return decision_variables

    def _poly_to_ommx(self, poly: amplify.Poly, constant: float = 0.0) -> Function:
        """
        Convert from the polynomial of the Fixstars Amplify SDK to the object of ommx.v1.
        """
        poly_dict = poly.as_dict()
        if poly.degree() <= 0:
            return Function(poly_dict.pop((), 0.0) - constant)
        elif poly.degree() == 1:
            constant = poly_dict.pop((), 0.0) - constant
            terms = {key[0]: value for key, value in poly_dict.items()}
            return Function(Linear(terms=terms, constant=constant))
        elif poly.degree() == 2:
            constant = poly_dict.pop((), 0.0) - constant
            columns = []
            rows = []
            values = []
            terms = {}
            for key, value in poly_dict.items():
                if len(key) == 2:
                    columns.append(key[0])
                    rows.append(key[1])
                    values.append(value)
                elif len(key) == 1:
                    terms[key[0]] = value
            return Function(
                Quadratic(
                    columns=columns,
                    rows=rows,
                    values=values,
                    linear=Linear(terms=terms, constant=constant),
                )
            )
        else:
            constant = poly_dict.pop((), 0.0) - constant
            terms = {}
            for key, value in poly_dict.items():
                terms[key] = value
            terms[()] = constant
            return Function(Polynomial(terms=terms))

    def objective(self) -> Function:
        if isinstance(self.model.objective, amplify.Matrix):
            objective = self.model.objective.to_poly()
        else:
            objective = self.model.objective
        return self._poly_to_ommx(objective)

    def constraints(self) -> typing.List[Constraint]:
        constraints = []
        counter = -1
        for constraint in self.model.constraints:
            counter += 1
            # Case: `amplify.less_than`
            if constraint.conditional[1] == "LE":
                assert isinstance(constraint.conditional[2], float)
                constraints.append(
                    Constraint(
                        id=counter,
                        function=self._poly_to_ommx(
                            constraint.conditional[0],
                            constraint.conditional[2],
                        ),
                        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                        name=constraint.label,
                    )
                )
            # Case: `amplify.equal_to`
            elif constraint.conditional[1] == "EQ":
                assert isinstance(constraint.conditional[2], float)
                constraints.append(
                    Constraint(
                        id=counter,
                        function=self._poly_to_ommx(
                            constraint.conditional[0],
                            constraint.conditional[2],
                        ),
                        equality=Constraint.EQUAL_TO_ZERO,
                        name=constraint.label,
                    )
                )
            # Case: `amplify.greater_than`
            elif constraint.conditional[1] == "GE":
                assert isinstance(constraint.conditional[2], float)
                # Convert to `LESS_THAN_OR_EQUAL_TO_ZERO` constraint.
                constraints.append(
                    Constraint(
                        id=counter,
                        function=self._poly_to_ommx(
                            -1 * constraint.conditional[0],
                            -1 * constraint.conditional[2],
                        ),
                        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                        name=constraint.label,
                    )
                )
            # Case: `amplify.clamp`
            elif constraint.conditional[1] == "BW":
                assert isinstance(constraint.conditional[2], tuple)
                # Split into two `LESS_THAN_OR_EQUAL_TO_ZERO` constraints.
                constraints.append(
                    Constraint(
                        id=counter,
                        function=self._poly_to_ommx(
                            -1 * constraint.conditional[0],
                            -1 * constraint.conditional[2][0],
                        ),
                        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                        name=constraint.label + "_lower",
                    )
                )
                counter += 1
                constraints.append(
                    Constraint(
                        id=counter,
                        function=self._poly_to_ommx(
                            constraint.conditional[0],
                            constraint.conditional[2][1],
                        ),
                        equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                        name=constraint.label + "_upper",
                    )
                )
            else:
                raise OMMXFixstarsAmplifyAdapterError(
                    f"Unintended constraint type: {constraint.conditional[1]}"
                )

        return constraints

    def sense(self):
        # NOTE:
        # According to the following link, the Fixstars Amplify SDK only supports
        # the `MINIMIZE` sense for the objective function. So return `MINIMIZE` here.
        # https://amplify.fixstars.com/ja/docs/amplify/v1/objective.html
        return Instance.MINIMIZE

    def _is_empty_model(self) -> bool:
        # NOTE:
        # A segment fault occurs when accessing the variables method of
        # the amplify.Model without decision variables.
        # So, without accessing this method, determine an empty mathematical model.
        if isinstance(self.model.objective, amplify.Poly):
            objective = self.model.objective
        else:
            objective = self.model.objective.to_poly()
        return objective.degree() <= 0 and len(self.model.constraints) == 0

    def build(self) -> Instance:
        if self._is_empty_model():
            # NOTE:
            # Note that even in the case of a non-zero constant objective function,
            # it will be returned as objective=0.
            return Instance.from_components(
                decision_variables=[],
                objective=0,
                constraints=[],
                sense=self.sense(),
            )
        else:
            return Instance.from_components(
                decision_variables=self.decision_variables(),
                objective=self.objective(),
                constraints=self.constraints(),
                sense=self.sense(),
            )


def model_to_instance(model: amplify.Model) -> Instance:
    """
    The function to create an ommx.v1.Instance from the Fixstars Amplify model.

    Example:
    =========
    The following example shows how to create an ommx.v1.Instance from a Fixstars Amplify model.

    .. doctest::

        >>> import amplify
        >>> from ommx_fixstars_amplify_adapter import model_to_instance
        >>>
        >>> gen = amplify.VariableGenerator()
        >>> x = gen.scalar("Binary", name="x")
        >>> model = amplify.Model(x)
        >>>
        >>> ommx_instance = model_to_instance(model)

    """
    builder = OMMXInstanceBuilder(model)
    return builder.build()
