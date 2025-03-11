import amplify

from typing import Union

from ommx.v1 import (
    Solution,
    Instance,
    DecisionVariable,
    Constraint,
    State,
)
from ommx.v1.function_pb2 import Function
from ommx.adapter import SolverAdapter
from ommx.v1.linear_pb2 import Linear
from ommx.v1.quadratic_pb2 import Quadratic
from ommx.v1.polynomial_pb2 import Polynomial
from ommx.v1.constraint_pb2 import Constraint as RawConstraint
from ommx.v1.decision_variables_pb2 import DecisionVariable as RawDVar

from .exception import OMMXFixstarsAmplifyAdapterError


class OMMXFixstarsAmplifyAdapter(SolverAdapter):
    def __init__(self, ommx_instance: Instance):
        """
        :param ommx_instance: The ommx.v1.Instance to solve.
        """
        self.instance = ommx_instance
        self.model = amplify.Model()

        self._set_decision_variables()
        self._set_objective()
        self._set_constraints()

    @classmethod
    def solve(
        cls, ommx_instance: Instance, *, amplify_token: str = "", timeout: int = 1000
    ) -> Solution:
        """Solve the given ommx.v1.Instance using Fixstars Amplify, returning an
        ommx.v1.Solution.

        **NOTE** The `amplify_token` parameter _must_ be passed to properly
          instantiate the Fixstars Client. Using the default value will result
          in an error.

        :param ommx_instance: The ommx.v1.Instance to solve.
        :param amplify_token: Token for instantiating the Fixstars Client, obtained from your Fixstars Amplify account.
        :param timeout: Timeout passed the client

        Example:
        =========
        The following example shows how to solve an unconstrained linear optimization problem with `x` as the objective function.

        .. doctest::

            >>> from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter
            >>> from ommx.v1 import Instance, DecisionVariable, Linear
            >>>
            >>> x1 = DecisionVariable.integer(1, lower=0, upper=5)
            >>> ommx_instance = Instance.from_components(
            ...     decision_variables=[x1],
            ...     objective=x1,
            ...     constraints=[],
            ...     sense=Instance.MINIMIZE,
            ... )
            >>> token = "YOUR API TOKEN" # Set your API token
            >>> solution = OMMXFixstarsAmplifyAdapter.solve(ommx_instance, amplify_token=token) # doctest: +SKIP
        """
        if amplify_token == "":
            raise OMMXFixstarsAmplifyAdapterError(
                "No Fixstars Amplify token specificed -- cannot instantiate client"
            )

        adapter = cls(ommx_instance)

        client = amplify.FixstarsClient()
        client.token = amplify_token
        client.parameters.timeout = timeout

        result = amplify.solve(adapter.solver_input, client)
        return adapter.decode(result)

    @property
    def solver_input(self) -> amplify.Model:
        """The Amplify model generated from this OMMX instance"""
        return self.model

    def decode(self, data: amplify.Result) -> Solution:
        """Convert optimized Python-MIP model and ommx.v1.Instance to ommx.v1.Solution.

        This method is intended to be used if the model has been acquired with
        `solver_input` for futher adjustment of the solver parameters, and
        separately optimizing the model.

        Note that alterations to the model may make the decoding process
        incompatible -- decoding will only work if the model still describes
        effectively the same problem as the OMMX instance used to create the
        adapter.

        Example:
        =========
        The following example shows how to solve an unconstrained linear optimization problem with `x` as the objective function.

        .. doctest::

            >>> from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter
            >>> from ommx.v1 import Instance, DecisionVariable, Linear
            >>>
            >>> x1 = DecisionVariable.integer(1, lower=0, upper=5)
            >>> ommx_instance = Instance.from_components(
            ...     decision_variables=[x1],
            ...     objective=x1,
            ...     constraints=[],
            ...     sense=Instance.MINIMIZE,
            ... )
            >>>
            >>> adapter = OMMXFixstarsAmplifyAdapter(ommx_instance)
            >>> model = adapter.solver_input
            >>> # ... some modification of model's parameters
            >>> client = amplify.FixstarsClient()
            >>> client.token = "YOUR API TOKEN" # Set your API token
            >>> client.parameters.timeout = 1000
            >>> result = amplify.solve(model, client)  # doctest: +SKIP
            >>> solution = adapter.decode(result)  # doctest: +SKIP
        """

        # TODO infeasible/unbounded detection
        state = self.decode_to_state(data)
        solution = self.instance.evaluate(state)

        return solution

    def decode_to_state(self, data: amplify.Result) -> State:
        """
        Create an ommx.v1.State from an amplify.Result.

        Example:
        =========
        The following example shows how to solve an unconstrained linear optimization problem with `x` as the objective function.

        .. doctest::

            >>> from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter
            >>> from ommx.v1 import Instance, DecisionVariable, Linear
            >>>
            >>> x1 = DecisionVariable.integer(1, lower=0, upper=5)
            >>> ommx_instance = Instance.from_components(
            ...     decision_variables=[x1],
            ...     objective=x1,
            ...     constraints=[],
            ...     sense=Instance.MINIMIZE,
            ... )
            >>>
            >>> adapter = OMMXFixstarsAmplifyAdapter(ommx_instance)
            >>> model = adapter.solver_input
            >>> # ... some modification of model's parameters
            >>> client = amplify.FixstarsClient()
            >>> client.token = "YOUR API TOKEN" # Set your API token
            >>> client.parameters.timeout = 1000
            >>> result = amplify.solve(model, client)  # doctest: +SKIP
            >>> state = adapter.decode_to_state(result)  # doctest: +SKIP
        """
        try:
            return State(
                entries={
                    key: value.evaluate(data.best.values)
                    for key, value in self.variable_map.items()
                }
            )
        except RuntimeError as e:
            raise OMMXFixstarsAmplifyAdapterError(
                f"Failed to create ommx.v1.State: {str(e)}"
            )

    def _set_decision_variables(self):
        self.variable_map = {}
        gen = amplify.VariableGenerator()
        for var in self.instance.raw.decision_variables:
            if var.kind == DecisionVariable.BINARY:
                amplify_var = gen.scalar(
                    "Binary",
                    name=_make_variable_label(var),
                )
            elif var.kind == DecisionVariable.INTEGER:
                amplify_var = gen.scalar(
                    "Integer",
                    bounds=(var.bound.lower, var.bound.upper),
                    name=_make_variable_label(var),
                )
            elif var.kind == DecisionVariable.CONTINUOUS:
                amplify_var = gen.scalar(
                    "Real",
                    bounds=(var.bound.lower, var.bound.upper),
                    name=_make_variable_label(var),
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
            if constr.equality == Constraint.EQUAL_TO_ZERO:
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
        self, func: Union[float, Linear, Quadratic, Polynomial, Function]
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


def _make_constraint_label(constraint: RawConstraint) -> str:
    return f"{constraint.name} [id: {constraint.id}]"


def _make_variable_label(variable: RawDVar) -> str:
    if len(variable.subscripts) == 0:
        return variable.name
    else:
        subscripts_str = "{" + ", ".join(map(str, variable.subscripts)) + "}"
        return f"{variable.name}_{subscripts_str}"
