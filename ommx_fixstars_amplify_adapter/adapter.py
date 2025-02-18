import amplify

import datetime


from ommx.v1 import (
    Solution,
    Instance,
    DecisionVariable,
    Linear,
    Quadratic,
    Polynomial,
    Constraint,
)
from ommx.v1.function_pb2 import Function
from ommx.adapter import SolverAdapter

from .exception import OMMXFixstarsAmplifyAdapterError


class OMMXFixstarsAmplifyAdapter(SolverAdapter):
    def __init__(self, ommx_instance: Instance):
        self.instance = ommx_instance
        self.model = amplify.Model()

        self._set_decision_variable()
        self._set_objective()
        self._set_constraints()

    @staticmethod
    def solve(
        ommx_instance: Instance, amplify_token: str, timeout: int = 1000
    ) -> Solution:
        adapter = OMMXFixstarsAmplifyAdapter(ommx_instance)

        client = amplify.FixstarsClient()
        client.token = amplify_token
        client.parameters.timeout = timeout

        result = amplify.solve(adapter.model, client)
        return adapter.decode(result)

    @property
    def solver_input(self) -> amplify.Model:
        return self.model

    def decode(self, data: amplify.Result) -> Solution:
        pass

    def _set_decision_variables(self):
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

    def _set_objective(self):
        obj_poly = self._function_to_poly(self.instance.raw.objective)
        if self.instance.raw.sense == Instance.MINIMIZE:
            self.model += obj_poly
        elif self.instance.raw.sense == Instance.MAXIMIZE:
            self.model += -obj_poly
        else:
            raise OMMXFixstarsAmplifyAdapterError(
                f"Unknown sense: {self.instance.raw.sense}"
            )

    def _set_constraints(self):
        for constr in self.instance.raw.constraints:
            function_poly = self._function_to_poly(constr.function)
            if constr.equality == Constraint.EQUALITY_EQUAL_TO_ZERO:
                self.model += amplify.equal_to(
                    function_poly, 0, label=_make_constraint_label(constr)
                )
            elif constr.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                self.model += amplify.less_equal(
                    function_poly, 0, label=_make_constraint_label(constr)
                )
            else:
                raise OMMXFixstarsAmplifyAdapterError(
                    f"Unknown equality type: {constr.equality}"
                )

    def _function_to_poly(
        self, func: float | Linear | Quadratic | Polynomial | Function
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


def _make_constraint_label(constraint: Constraint) -> str:
    return f"{constraint.name} [id: {constraint.id}]"
