import os
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

import amplify
from ommx.v1 import Instance

from ommx_fixstars_amplify_adapter import OMMXFixstarsAmplifyAdapter

from instance import (
    build_assignment_instance,
    build_blending_instance,
    build_facility_location_instance,
    build_knapsack_instance,
    build_portfolio_instance,
    build_production_instance,
    build_tsp_instance,
)

INSTANCE_BUILDERS = {
    "knapsack": build_knapsack_instance,
    "production": build_production_instance,
    "blending": build_blending_instance,
    "assignment": build_assignment_instance,
    "facility-location": build_facility_location_instance,
    "portfolio": build_portfolio_instance,
    "tsp": build_tsp_instance,
}
INSTANCE_NAMES = tuple(INSTANCE_BUILDERS)
FORMULATIONS = ("regular", "one-hot")

PACKAGE_VERSIONS = (
    version("ommx"),
    version("amplify"),
    version("ommx_fixstars_amplify_adapter"),
)


def build_instance(name: str, size: int, seed: int, formulation: str) -> Instance:
    """Select and build a benchmark Instance."""
    return INSTANCE_BUILDERS[name](size, seed, formulation)


def prepare_target(
    operation: str, instance: Instance, solver_time_limit_ms: int
) -> Callable[[], Any]:
    """Prepare everything outside the measured operation."""
    if operation == "instance-to-model":
        return lambda: OMMXFixstarsAmplifyAdapter(instance).solver_input

    token = os.environ.get("AMPLIFY_TOKEN")
    if not token:
        raise RuntimeError("AMPLIFY_TOKEN is required")

    adapter = OMMXFixstarsAmplifyAdapter(instance)
    client = amplify.AmplifyAEClient()  # pyright: ignore[reportAttributeAccessIssue]
    client.token = token
    client.parameters.time_limit_ms = solver_time_limit_ms
    result = amplify.solve(adapter.solver_input, client)
    return lambda: adapter.decode(result)
