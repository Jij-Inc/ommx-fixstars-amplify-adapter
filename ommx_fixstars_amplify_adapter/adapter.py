import copy
from typing import ClassVar

import amplify

from ommx import (
    Solution,
    Instance,
    DecisionVariable,
    Constraint,
    Function,
    State,
    DegreeBound,
    Equality,
    InstanceClass,
    InstanceClassClause,
    Kind,
    PreparationPolicy,
    Sense,
    SpecialConstraintKind,
    SpecialConstraintPreparation,
)
from ommx.adapter import DiagnosticsSink, SolverAdapter

from .exception import OMMXFixstarsAmplifyAdapterError


ABSOLUTE_TOLERANCE = 1e-6


class OMMXFixstarsAmplifyAdapter(SolverAdapter):
    INPUT_CLASS: ClassVar[InstanceClass] = InstanceClass(
        [
            InstanceClassClause(
                label="fixstars-amplify-polynomial",
                allowed_variable_kinds={
                    Kind.Binary,
                    Kind.Integer,
                    Kind.Continuous,
                },
                objective_degree_bound=DegreeBound.unbounded(),
                regular_constraint_degree_bounds={
                    Equality.EqualToZero: DegreeBound.unbounded(),
                    Equality.LessThanOrEqualToZero: DegreeBound.unbounded(),
                },
                allows_one_hot=True,
                allowed_senses={Sense.Minimize, Sense.Maximize},
            )
        ]
    )

    @classmethod
    def recommended_preparation_policy(cls) -> PreparationPolicy:
        """Recommend lowering unsupported special constraints before using Amplify.

        Amplify accepts OneHot constraints directly, so this recommendation
        preserves them and lowers only Indicator and SOS1 constraints. The
        returned policy is fresh and caller-editable.
        """
        return PreparationPolicy(
            special_constraints=SpecialConstraintPreparation.lower_special_constraints(
                kinds={
                    SpecialConstraintKind.Indicator,
                    SpecialConstraintKind.Sos1,
                }
            )
        )

    def __init__(self, ommx_instance: Instance):
        """
        :param ommx_instance: The ommx.Instance to solve.
        """
        self.require_applicable(ommx_instance)
        self.instance = ommx_instance
        self.model = amplify.Model()

        self._set_decision_variables()
        self._set_objective()
        self._set_constraints()

    @classmethod
    def solve(
        cls,
        ommx_instance: Instance,
        *,
        amplify_token: str = "",
        timeout: int = 1000,
        diagnostics: DiagnosticsSink | None = None,
    ) -> Solution:
        """Solve the given ommx.Instance using Fixstars Amplify AE, returning an
        ommx.Solution.

        ``diagnostics`` are not available through this Adapter.
        The reserved ``diagnostics`` argument is accepted for compatibility with
        the OMMX SolverAdapter interface.

        **NOTE** The `amplify_token` parameter _must_ be passed to properly
          instantiate the Fixstars Amplify AE Client. Using the default value will result
          in an error.

        :param ommx_instance: The ommx.Instance to prepare and solve.
        :param amplify_token: Token for instantiating the Fixstars Amplify AE Client, obtained from your Fixstars Amplify account.
        :param timeout: Timeout passed to the client.
        :param diagnostics: Reserved for OMMX SolverAdapter compatibility;
          currently unused.

        Example:
        =========
        The following example shows how to solve an unconstrained linear optimization problem with `x` as the objective function.

        .. doctest::

            >>> from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter
            >>> from ommx import Instance, DecisionVariable
            >>>
            >>> x1 = DecisionVariable.integer(1, lower=0, upper=5)
            >>> ommx_instance = Instance.from_components(
            ...     decision_variables=[x1],
            ...     objective=x1,
            ...     constraints={},
            ...     sense=Instance.MINIMIZE,
            ... )
            >>> token = "YOUR API TOKEN" # Set your API token
            >>> solution = OMMXFixstarsAmplifyAdapter.solve(ommx_instance, amplify_token=token) # doctest: +SKIP
        """
        prepared = copy.copy(ommx_instance)
        prepared.prepare(cls.INPUT_CLASS, cls.recommended_preparation_policy())
        return cls.solve_without_preparation(
            prepared,
            amplify_token=amplify_token,
            timeout=timeout,
            diagnostics=diagnostics,
        )

    @classmethod
    def solve_without_preparation(
        cls,
        ommx_instance: Instance,
        *,
        amplify_token: str = "",
        timeout: int = 1000,
        diagnostics: DiagnosticsSink | None = None,
    ) -> Solution:
        """Solve an exact Fixstars Amplify Adapter input without preparing it.

        Use this method when the input instance has already been prepared,
        possibly with a custom policy, or already belongs to ``INPUT_CLASS``.

        ``diagnostics`` are not available through this Adapter.
        The reserved ``diagnostics`` argument is accepted for compatibility with
        the OMMX SolverAdapter interface.

        **NOTE** The ``amplify_token`` parameter *must* be passed to properly
          instantiate the Fixstars Amplify AE Client. Using the default value will
          result in an error.

        :param ommx_instance: The exact Fixstars Amplify Adapter input to solve.
        :param amplify_token: Token for instantiating the Fixstars Amplify AE
          Client, obtained from your Fixstars Amplify account.
        :param timeout: Timeout passed to the client.
        :param diagnostics: Reserved for OMMX SolverAdapter compatibility;
          currently unused.
        """
        if amplify_token == "":
            raise OMMXFixstarsAmplifyAdapterError(
                "No Fixstars Amplify token specified -- cannot instantiate client"
            )

        # TODO: Update the diagnostics docstrings when support is implemented.
        _ = diagnostics
        adapter = cls(ommx_instance)

        client = amplify.AmplifyAEClient()
        client.token = amplify_token
        client.parameters.time_limit_ms = timeout

        result = amplify.solve(adapter.solver_input, client)
        return adapter.decode(result)

    @property
    def solver_input(self) -> amplify.Model:
        """The Amplify model generated from this OMMX instance"""
        return self.model

    def decode(self, data: amplify.Result) -> Solution:
        """Convert Amplify result and ommx.Instance to ommx.Solution.

        This method is intended to be used if the model has been acquired with
        `solver_input` for further adjustment of the solver parameters, and
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
            >>> from ommx import Instance, DecisionVariable
            >>>
            >>> x1 = DecisionVariable.integer(1, lower=0, upper=5)
            >>> ommx_instance = Instance.from_components(
            ...     decision_variables=[x1],
            ...     objective=x1,
            ...     constraints={},
            ...     sense=Instance.MINIMIZE,
            ... )
            >>>
            >>> adapter = OMMXFixstarsAmplifyAdapter(ommx_instance)
            >>> model = adapter.solver_input
            >>> # ... some modification of model's parameters
            >>> client = amplify.AmplifyAEClient()
            >>> client.token = "YOUR API TOKEN" # Set your API token
            >>> client.parameters.time_limit_ms = 1000
            >>> result = amplify.solve(model, client)  # doctest: +SKIP
            >>> solution = adapter.decode(result)  # doctest: +SKIP
        """

        # TODO infeasible/unbounded detection
        state = self.decode_to_state(data)
        solution = self.instance.evaluate(state)

        return solution

    def decode_to_state(self, data: amplify.Result) -> State:
        """
        Create an ommx.State from an amplify.Result.

        Example:
        =========
        The following example shows how to solve an unconstrained linear optimization problem with `x` as the objective function.

        .. doctest::

            >>> from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter
            >>> from ommx import Instance, DecisionVariable
            >>>
            >>> x1 = DecisionVariable.integer(1, lower=0, upper=5)
            >>> ommx_instance = Instance.from_components(
            ...     decision_variables=[x1],
            ...     objective=x1,
            ...     constraints={},
            ...     sense=Instance.MINIMIZE,
            ... )
            >>>
            >>> adapter = OMMXFixstarsAmplifyAdapter(ommx_instance)
            >>> model = adapter.solver_input
            >>> # ... some modification of model's parameters
            >>> client = amplify.AmplifyAEClient()
            >>> client.token = "YOUR API TOKEN" # Set your API token
            >>> client.parameters.time_limit_ms = 1000
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
                f"Failed to create ommx.State: {str(e)}"
            )

    def _set_decision_variables(self):
        self.variable_map: dict[int, amplify.Poly] = {}
        gen = amplify.VariableGenerator()
        for var in self.instance.used_decision_variables:
            if var.kind == DecisionVariable.BINARY:
                self.variable_map[var.id] = gen.scalar(
                    amplify.VariableType.Binary,
                    name=_make_variable_label(var),
                )
            elif var.kind == DecisionVariable.INTEGER:
                self.variable_map[var.id] = gen.scalar(
                    amplify.VariableType.Integer,
                    bounds=(var.bound.lower, var.bound.upper),
                    name=_make_variable_label(var),
                )
            elif var.kind == DecisionVariable.CONTINUOUS:
                self.variable_map[var.id] = gen.scalar(
                    amplify.VariableType.Real,
                    bounds=(var.bound.lower, var.bound.upper),
                    name=_make_variable_label(var),
                )
            else:
                raise AssertionError(
                    "Unsupported decision variable kind reached after applicability "
                    f"validation: {var.kind}. This may indicate an OMMX implementation "
                    "bug; please report it to OMMX."
                )

    def _set_objective(self):
        obj_poly = self._function_to_poly(self.instance.objective)
        if self.instance.sense == Instance.MINIMIZE:
            self.model += obj_poly
        elif self.instance.sense == Instance.MAXIMIZE:
            self.model += -obj_poly
        else:
            raise AssertionError(
                "Unsupported objective sense reached after applicability validation: "
                f"{self.instance.sense}. This may indicate an OMMX implementation "
                "bug; please report it to OMMX."
            )

    def _set_constraints(self):
        # Handle one_hot constraints
        for one_hot_id, one_hot in self.instance.one_hot_constraints.items():
            # convert one_hot constraint to polynomial
            one_hot_poly = amplify.sum(
                self.variable_map[var_id] for var_id in one_hot.variables
            )
            self.model += amplify.one_hot(
                one_hot_poly, label=f"{one_hot.name} [id: {one_hot_id}]"
            )

        supported_equalities = {
            Constraint.EQUAL_TO_ZERO,
            Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
        }
        for constr_id, constr in self.instance.constraints.items():
            if constr.equality not in supported_equalities:
                raise AssertionError(
                    "Unsupported constraint equality reached after applicability "
                    f"validation: {constr.equality} for constraint {constr_id}. This "
                    "may indicate an OMMX implementation bug; please report it to OMMX."
                )

            if constr.function.degree() == 0:
                if constr.evaluate({}, atol=ABSOLUTE_TOLERANCE).feasible:
                    continue
                raise OMMXFixstarsAmplifyAdapterError(
                    f"Infeasible constant constraint was found: id {constr_id}"
                )

            function_poly = self._function_to_poly(constr.function)
            if constr.equality == Constraint.EQUAL_TO_ZERO:
                self.model += amplify.equal_to(
                    function_poly, 0, label=f"{constr.name} [id: {constr_id}]"
                )
            elif constr.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                self.model += amplify.less_equal(
                    function_poly, 0, label=f"{constr.name} [id: {constr_id}]"
                )

    def _function_to_poly(
        self,
        func: Function,
    ) -> amplify.Poly:
        poly = amplify.Poly(0)
        for ids, coefficient in func.terms.items():
            if len(ids) == 0:
                poly += coefficient
            else:
                term = coefficient
                for id in ids:
                    term *= self.variable_map[id]
                poly += term
        return poly


def _make_variable_label(variable: DecisionVariable) -> str:
    if len(variable.subscripts) == 0:
        return variable.name
    else:
        subscripts_str = "{" + ", ".join(map(str, variable.subscripts)) + "}"
        return f"{variable.name}_{subscripts_str}"
