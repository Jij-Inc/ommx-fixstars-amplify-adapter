import copy
from typing import ClassVar, final

import amplify

from ommx import (
    Solution,
    Instance,
    DecisionVariable,
    Constraint,
    ConstraintKind,
    Function,
    InfeasibleDetected,
    State,
    Equality,
    InstanceClass,
    InstanceClassClause,
    Kind,
    PolynomialRequirement,
    PreparationPolicy,
    Sense,
    SpecialConstraintPreparation,
)
from ommx.adapter import DiagnosticsSink, NoSolutionReturned, SolverAdapter

from .exception import OMMXFixstarsAmplifyAdapterError


ABSOLUTE_TOLERANCE = 1e-6


@final
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
                objective_polynomial_requirement=PolynomialRequirement.any_degree(),
                regular_constraint_polynomial_requirements={
                    Equality.EqualToZero: PolynomialRequirement.any_degree(),
                    Equality.LessThanOrEqualToZero: PolynomialRequirement.any_degree(),
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
                    ConstraintKind.Indicator,
                    ConstraintKind.Sos1,
                }
            )
        )

    def __init__(self, ommx_instance: Instance):
        """Build the Amplify model and retain the supplied OMMX instance by reference.

        Keep the instance unchanged from construction until you finish using the
        adapter, including backend configuration, execution, and decoding. To edit
        the original during that time, pass ``copy.copy(instance)`` to the constructor.

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

        Preparation runs on a private copy of the supplied instance.

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
            >>> from ommx import Instance
            >>>
            >>> ommx_instance = Instance.minimize()
            >>> x1 = ommx_instance.new_integer("x1", lower=0, upper=5)
            >>> ommx_instance.objective = x1
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
        Keep the supplied instance unchanged until this call returns.

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
        adapter = cls(ommx_instance)
        if amplify_token == "":
            raise OMMXFixstarsAmplifyAdapterError(
                "No Fixstars Amplify token specified -- cannot instantiate client"
            )

        # TODO: Update the diagnostics docstrings when support is implemented.
        _ = diagnostics

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

        Decoding evaluates the state against the OMMX instance passed to the
        constructor. Keep that instance unchanged while using the adapter.
        Any changes to the Amplify model must preserve its correspondence with
        the OMMX problem for decoding to remain valid.

        :raises ommx.adapter.NoSolutionReturned: If there are active variables
            to decode and Amplify returned no candidates.

        Example:
        =========
        The following example shows how to solve an unconstrained linear optimization problem with `x` as the objective function.

        .. doctest::

            >>> from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter
            >>> from ommx import Instance
            >>>
            >>> ommx_instance = Instance.minimize()
            >>> x1 = ommx_instance.new_integer("x1", lower=0, upper=5)
            >>> ommx_instance.objective = x1
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
            >>> from ommx import Instance
            >>>
            >>> ommx_instance = Instance.minimize()
            >>> x1 = ommx_instance.new_integer("x1", lower=0, upper=5)
            >>> ommx_instance.objective = x1
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
        if not self.variable_map:
            return State(entries={})
        if len(data) == 0:
            raise NoSolutionReturned("Amplify returned no candidates.")
        try:
            values = data.best.values
            return State(
                entries={
                    var_id: poly.evaluate(values)
                    for var_id, poly in self.variable_map.items()
                }
            )
        except RuntimeError as e:
            raise OMMXFixstarsAmplifyAdapterError(
                f"Failed to create ommx.State: {str(e)}"
            )

    def _set_decision_variables(self):
        self.variable_map: dict[int, amplify.Poly] = {}
        gen = amplify.VariableGenerator()
        for var in self.instance.active_decision_variables:
            bound = var.bound
            if var.kind == DecisionVariable.BINARY and bound.lower != bound.upper:
                self.variable_map[var.id] = gen.scalar(
                    amplify.VariableType.Binary,
                    name=_make_variable_label(var),
                )
            elif var.kind in (DecisionVariable.BINARY, DecisionVariable.INTEGER):
                # Amplify represents a fixed binary domain as a bounded Integer.
                self.variable_map[var.id] = gen.scalar(
                    amplify.VariableType.Integer,
                    bounds=(bound.lower, bound.upper),
                    name=_make_variable_label(var),
                )
            elif var.kind == DecisionVariable.CONTINUOUS:
                self.variable_map[var.id] = gen.scalar(
                    amplify.VariableType.Real,
                    bounds=(bound.lower, bound.upper),
                    name=_make_variable_label(var),
                )
            else:
                raise AssertionError(
                    "Unsupported decision variable kind reached after applicability "
                    f"validation: {var.kind}. This may indicate an OMMX implementation "
                    "bug; please report it to OMMX."
                )

    def _set_objective(self):
        objective = self.instance.objective
        if objective.degree() is None:
            raise AssertionError(
                "Non-polynomial objective reached after applicability validation. "
                "This may indicate an OMMX implementation bug; please report it to "
                "OMMX."
            )

        obj_poly = self._function_to_poly(objective)
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
        unsupported = self.instance.active_constraint_kinds - {
            ConstraintKind.Regular,
            ConstraintKind.OneHot,
        }
        if unsupported:
            raise AssertionError(
                f"Unsupported active constraint families after applicability validation: {unsupported}"
            )
        active = self.instance.active_constraints
        for one_hot_id, one_hot in active.of_kind(ConstraintKind.OneHot).items():
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
        for constr_id, constr in active.of_kind(ConstraintKind.Regular).items():
            if constr.equality not in supported_equalities:
                raise AssertionError(
                    "Unsupported constraint equality reached after applicability "
                    f"validation: {constr.equality} for constraint {constr_id}. This "
                    "may indicate an OMMX implementation bug; please report it to OMMX."
                )

            constraint_degree = constr.function.degree()
            if constraint_degree is None:
                raise AssertionError(
                    "Non-polynomial constraint reached after applicability validation: "
                    f"constraint {constr_id}. This may indicate an OMMX implementation "
                    "bug; please report it to OMMX."
                )

            if constraint_degree == 0:
                evaluated = constr.evaluate({}, atol=ABSOLUTE_TOLERANCE)
                if evaluated.is_feasible(atol=ABSOLUTE_TOLERANCE):
                    continue
                raise InfeasibleDetected(
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
