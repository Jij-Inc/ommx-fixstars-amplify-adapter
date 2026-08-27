import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import amplify

if __package__:
    from . import instance as benchmark_instances
else:
    import instance as benchmark_instances
from ommx import Instance

from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter

INSTANCE_BUILDERS = {
    "knapsack": benchmark_instances.build_knapsack_instance,
    "production": benchmark_instances.build_production_instance,
    "blending": benchmark_instances.build_blending_instance,
    "assignment": benchmark_instances.build_assignment_instance,
    "facility-location": benchmark_instances.build_facility_location_instance,
    "portfolio": benchmark_instances.build_portfolio_instance,
    "portfolio-cardinality": benchmark_instances.build_portfolio_cardinality_instance,
    "unit-commitment": benchmark_instances.build_unit_commitment_instance,
    "clique": benchmark_instances.build_clique_instance,
    "tsp": benchmark_instances.build_tsp_instance,
}
PREPARATION_INSTANCE_NAME = "one-hot-preparation"
INSTANCE_NAMES = (*INSTANCE_BUILDERS, PREPARATION_INSTANCE_NAME)
FORMULATIONS = ("regular", "one-hot")
SPECIAL_CONSTRAINT_CASES = ("none", "indicator", "sos1", "indicator-sos1")
PREPARATIONS = ("none", "recommended")
OPERATIONS = (
    "prepare",
    "instance-to-model",
    "result-to-solution",
    "end-to-end",
)

PACKAGE_VERSIONS = (
    version("ommx"),
    version("amplify"),
    version("ommx_fixstars_amplify_adapter"),
)


@dataclass(frozen=True)
class BenchmarkOperation:
    """Separate per-sample setup from the operation being measured."""

    setup: Callable[[], Any]
    run: Callable[[Any], Any]


def build_instance(
    name: str,
    size: int,
    seed: int,
    formulation: str,
    special_constraints: str = "none",
    preparation: str = "none",
) -> Instance:
    """Select and build a benchmark Instance."""
    if name == PREPARATION_INSTANCE_NAME:
        return benchmark_instances.build_one_hot_preparation_instance(
            size,
            seed,
            formulation,
            special_constraints,
            preparation,
        )
    if special_constraints != "none":
        raise ValueError(
            "Special constraints are available only for one-hot-preparation"
        )
    if preparation != "none":
        raise ValueError(
            "Preparation is available only for one-hot-preparation special constraints"
        )
    return INSTANCE_BUILDERS[name](size, seed, formulation)


def _prepare_instance(instance: Instance) -> Instance:
    input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS
    prepared = copy.copy(instance)
    prepared.prepare(
        input_class,
        OMMXFixstarsAmplifyAdapter.recommended_preparation_policy(),
    )
    return prepared


def make_benchmark_operation(
    operation: str,
    instance: Instance,
    solver_time_limit_ms: int,
    special_constraints: str,
    preparation: str,
) -> BenchmarkOperation:
    """Prepare setup and measured call for a benchmark operation."""
    if operation == "prepare":
        if special_constraints == "none" or preparation != "recommended":
            raise ValueError(
                "prepare requires Indicator and/or SOS1 constraints with "
                "recommended preparation"
            )
        input_class = OMMXFixstarsAmplifyAdapter.INPUT_CLASS

        def setup_preparation() -> tuple[Instance, Any]:
            return (
                copy.copy(instance),
                OMMXFixstarsAmplifyAdapter.recommended_preparation_policy(),
            )

        def run_preparation(context: tuple[Instance, Any]) -> Instance:
            prepared, policy = context
            prepared.prepare(input_class, policy)
            return prepared

        return BenchmarkOperation(setup=setup_preparation, run=run_preparation)

    if operation == "end-to-end":
        token = os.environ.get("AMPLIFY_TOKEN")
        if not token:
            raise RuntimeError("AMPLIFY_TOKEN is required")
        if preparation == "recommended":
            return BenchmarkOperation(
                setup=lambda: instance,
                run=lambda target: OMMXFixstarsAmplifyAdapter.solve(
                    target,
                    amplify_token=token,
                    timeout=solver_time_limit_ms,
                ),
            )
        return BenchmarkOperation(
            setup=lambda: instance,
            run=lambda target: OMMXFixstarsAmplifyAdapter.solve_without_preparation(
                target,
                amplify_token=token,
                timeout=solver_time_limit_ms,
            ),
        )

    adapter_instance = (
        _prepare_instance(instance) if preparation == "recommended" else instance
    )
    if operation == "instance-to-model":
        return BenchmarkOperation(
            setup=lambda: adapter_instance,
            run=lambda target: OMMXFixstarsAmplifyAdapter(target).solver_input,
        )

    if operation != "result-to-solution":
        raise ValueError(f"Unknown operation: {operation}")

    token = os.environ.get("AMPLIFY_TOKEN")
    if not token:
        raise RuntimeError("AMPLIFY_TOKEN is required")

    adapter = OMMXFixstarsAmplifyAdapter(adapter_instance)
    client = amplify.AmplifyAEClient()  # pyright: ignore[reportAttributeAccessIssue]
    client.token = token
    client.parameters.time_limit_ms = solver_time_limit_ms
    result = amplify.solve(adapter.solver_input, client)
    return BenchmarkOperation(
        setup=lambda: result,
        run=lambda solver_result: adapter.decode(solver_result),
    )
