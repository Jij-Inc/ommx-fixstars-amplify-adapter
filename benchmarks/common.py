import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import amplify
from ommx.v1 import Instance

from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter

from instance import (
    build_assignment_instance,
    build_blending_instance,
    build_clique_instance,
    build_facility_location_instance,
    build_knapsack_instance,
    build_one_hot_preparation_instance,
    build_portfolio_cardinality_instance,
    build_portfolio_instance,
    build_production_instance,
    build_tsp_instance,
    build_unit_commitment_instance,
)

INSTANCE_BUILDERS = {
    "knapsack": build_knapsack_instance,
    "production": build_production_instance,
    "blending": build_blending_instance,
    "assignment": build_assignment_instance,
    "facility-location": build_facility_location_instance,
    "portfolio": build_portfolio_instance,
    "portfolio-cardinality": build_portfolio_cardinality_instance,
    "unit-commitment": build_unit_commitment_instance,
    "clique": build_clique_instance,
    "tsp": build_tsp_instance,
    "one-hot-preparation": build_one_hot_preparation_instance,
}
INSTANCE_NAMES = tuple(INSTANCE_BUILDERS)
FORMULATIONS = ("regular", "one-hot")
SPECIAL_CONSTRAINT_CASES = ("none", "indicator", "sos1", "indicator-sos1")

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
    if special_constraints != "none":
        raise ValueError("OMMX v2 does not support Indicator or SOS1 constraints")
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

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    client = amplify.AmplifyAEClient()  # pyright: ignore[reportAttributeAccessIssue]
    client.token = token
    client.parameters.time_limit_ms = solver_time_limit_ms
    result = amplify.solve(adapter.solver_input, client)
    return BenchmarkOperation(
        setup=lambda: result,
        run=lambda solver_result: adapter.decode(solver_result),
    )
