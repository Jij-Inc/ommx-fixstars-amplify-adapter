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
from ommx.v1 import Instance

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
    "one-hot-preparation": benchmark_instances.build_one_hot_preparation_instance,
}
INSTANCE_NAMES = tuple(INSTANCE_BUILDERS)
FORMULATIONS = ("regular", "one-hot")
SPECIAL_CONSTRAINT_CASES = ("none", "indicator", "sos1", "indicator-sos1")
OPERATIONS = ("instance-to-model", "result-to-solution", "end-to-end")

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
) -> Instance:
    """Select and build a benchmark Instance."""
    if name == "one-hot-preparation":
        return benchmark_instances.build_one_hot_preparation_instance(
            size,
            seed,
            formulation,
            special_constraints,
        )
    if special_constraints != "none":
        raise ValueError(
            "Special constraint counterparts are available only for one-hot-preparation"
        )
    return INSTANCE_BUILDERS[name](size, seed, formulation)


def make_benchmark_operation(
    operation: str, instance: Instance, solver_time_limit_ms: int
) -> BenchmarkOperation:
    """Prepare everything outside the measured operation."""
    if operation == "instance-to-model":
        return BenchmarkOperation(
            setup=lambda: instance,
            run=lambda target: OMMXFixstarsAmplifyAdapter(target).solver_input,
        )

    token = os.environ.get("AMPLIFY_TOKEN")
    if not token:
        raise RuntimeError("AMPLIFY_TOKEN is required")

    if operation == "end-to-end":
        return BenchmarkOperation(
            setup=lambda: instance,
            run=lambda target: OMMXFixstarsAmplifyAdapter.solve(
                target,
                amplify_token=token,
                timeout=solver_time_limit_ms,
            ),
        )

    if operation != "result-to-solution":
        raise ValueError(f"Unknown operation: {operation}")

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    client = amplify.AmplifyAEClient()  # pyright: ignore[reportAttributeAccessIssue]
    client.token = token
    client.parameters.time_limit_ms = solver_time_limit_ms
    result = amplify.solve(adapter.solver_input, client)
    return BenchmarkOperation(
        setup=lambda: result,
        run=lambda solver_result: adapter.decode(solver_result),
    )
