import typing
from dataclasses import dataclass, field

import amplify
from ommx.v1 import Instance
from ommx.v1.constraint_pb2 import Constraint, Equality
from ommx.v1.decision_variables_pb2 import DecisionVariable
from ommx.v1.function_pb2 import Function
from ommx.v1.linear_pb2 import Linear
from ommx.v1.quadratic_pb2 import Quadratic
from ommx.v1.polynomial_pb2 import Polynomial

from .exception import OMMXFixstarsAmplifyAdapterError


@dataclass
class AmplifyModelBuilder:
    """
    Build an Amplify Model from an ommx.v1.Instance.
    """

    instance: Instance
    variable_map: typing.Dict[int, amplify.Poly] = field(default_factory=dict)

    def variables(self) -> typing.Dict[int, amplify.Poly]:
        self.variable_map = {}
        gen = amplify.VariableGenerator()
        for var in self.instance.raw.decision_variables:
            if var.kind == DecisionVariable.Kind.KIND_BINARY:
                amplify_var = gen.scalar(
                    "Binary",
                    name=self._make_variable_label(var),
                )
            elif var.kind == DecisionVariable.Kind.KIND_INTEGER:
                amplify_var = gen.scalar(
                    "Integer",
                    bounds=(var.bound.lower, var.bound.upper),
                    name=self._make_variable_label(var),
                )
            elif var.kind == DecisionVariable.Kind.KIND_CONTINUOUS:
                amplify_var = gen.scalar(
                    "Real",
                    bounds=(var.bound.lower, var.bound.upper),
                    name=self._make_variable_label(var),
                )
            else:
                raise OMMXFixstarsAmplifyAdapterError(
                    f"Not supported decision variable kind: {var.kind}"
                )
            self.variable_map[var.id] = amplify_var
        return self.variable_map

    def _function_to_poly(
        self, func: typing.Union[float, Linear, Quadratic, Polynomial, Function]
    ) -> amplify.Poly:
        if isinstance(func, (float, int)):
            return amplify.Poly(float(func))

        elif isinstance(func, Linear):
            poly = amplify.Poly(func.constant)
            for term in func.terms:
                var = self.variable_map[term.id]
                poly += term.coefficient * var
            return poly

        elif isinstance(func, Quadratic):
            poly = amplify.Poly()
            for col, row, value in zip(func.columns, func.rows, func.values):
                var_col = self.variable_map[col]
                var_row = self.variable_map[row]
                poly += value * var_col * var_row
            poly += self._function_to_poly(func.linear)
            return poly

        elif isinstance(func, Polynomial):
            poly = amplify.Poly()
            for monomial in func.terms:
                term = monomial.coefficient
                for var_id in monomial.ids:
                    term *= self.variable_map[var_id]
                poly += term
            return poly

        elif isinstance(func, Function):
            if func.HasField("constant"):
                return amplify.Poly(func.constant)
            if func.HasField("linear"):
                return self._function_to_poly(func.linear)
            elif func.HasField("quadratic"):
                return self._function_to_poly(func.quadratic)
            elif func.HasField("polynomial"):
                return self._function_to_poly(func.polynomial)
            else:
                raise OMMXFixstarsAmplifyAdapterError(
                    f"Unknown fields in Function: {func}"
                )

        else:
            raise OMMXFixstarsAmplifyAdapterError(
                f"Unknown function type: {type(func)}"
            )

    def _make_variable_label(self, ommx_variable: DecisionVariable) -> str:
        if len(ommx_variable.subscripts) == 0:
            return ommx_variable.name
        else:
            subscripts_str = "{" + ", ".join(map(str, ommx_variable.subscripts)) + "}"
            return f"{ommx_variable.name}_{subscripts_str}"

    def _make_constraint_label(self, ommx_constraint: Constraint) -> str:
        return f"{ommx_constraint.name} [id: {ommx_constraint.id}]"

    def objective(self) -> amplify.Poly:
        return self._function_to_poly(self.instance.raw.objective)

    def constraints(self) -> typing.List:
        constraints = []
        for ommx_constraint in self.instance.raw.constraints:
            function_poly = self._function_to_poly(ommx_constraint.function)
            if ommx_constraint.equality == Equality.EQUALITY_EQUAL_TO_ZERO:
                constraints.append(
                    amplify.equal_to(
                        function_poly,
                        0,
                        label=self._make_constraint_label(ommx_constraint),
                    )
                )
            elif (
                ommx_constraint.equality == Equality.EQUALITY_LESS_THAN_OR_EQUAL_TO_ZERO
            ):
                constraints.append(
                    amplify.less_equal(
                        function_poly,
                        0,
                        label=self._make_constraint_label(ommx_constraint),
                    )
                )
            else:
                raise OMMXFixstarsAmplifyAdapterError(
                    f"Unknown equality type: {ommx_constraint.equality}"
                )

        return constraints

    def sense(self):
        return self.instance.raw.sense

    def build(self) -> typing.Tuple[amplify.Model, typing.Dict[int, amplify.Poly]]:
        variable_map = self.variables()
        objective_poly = self.objective()
        constraints = self.constraints()

        model = amplify.Model()
        if self.sense() == Instance.MINIMIZE:
            model += objective_poly
        elif self.sense() == Instance.MAXIMIZE:
            model += -objective_poly
        else:
            raise OMMXFixstarsAmplifyAdapterError(f"Unknown sense: {self.sense()}")

        for constraint_expr in constraints:
            model += constraint_expr

        return model, variable_map


def instance_to_model(
    instance: Instance,
) -> typing.Tuple[amplify.Model, typing.Dict[int, amplify.Poly]]:
    """
    The function to create an Amplify model and variable map from an ommx.v1.Instance.
    The variable map is a dictionary that links the ID of each decision variable in the OMMX instance to its corresponding variable in the Amplify model.

    Example:
    =========
    The following example shows how to create a Fixstars Amplify model from an ommx.v1.Instance.

    .. doctest::

        >>> import amplify
        >>> from ommx_fixstars_amplify_adapter import instance_to_model
        >>> from ommx.v1 import Instance, DecisionVariable, Linear
        >>>
        >>> x_var = DecisionVariable.of_type(kind=DecisionVariable.BINARY, id=0, name='x', lower=0, upper=1)
        >>> objective = Linear(terms={0: 1.0}, constant=0.0)
        >>> instance = Instance.from_components(decision_variables=[x_var], objective=objective, constraints=[], sense=Instance.MINIMIZE)
        >>>
        >>> model, variable_map = instance_to_model(instance)

    """
    builder = AmplifyModelBuilder(instance)
    return builder.build()
