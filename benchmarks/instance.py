import math
import random

from ommx.v1 import (
    Constraint,
    ConstraintHints,
    DecisionVariable,
    Instance,
    Linear,
    OneHot,
    Quadratic,
)


def _check_size(size: int, minimum: int = 1) -> None:
    if size < minimum:
        raise ValueError(f"Size must be at least {minimum}")


def _require_regular(formulation: str) -> None:
    if formulation != "regular":
        raise ValueError("This Instance only supports the regular formulation")


def _build_one_hot_constraints(
    specs: list[tuple[str, list[int], list[int]]], formulation: str
) -> tuple[list[Constraint], ConstraintHints | None]:
    if formulation not in ("regular", "one-hot"):
        raise ValueError(f"Unknown formulation: {formulation}")

    constraints = [
        Constraint(
            id=constraint_id,
            function=Linear(
                terms={variable_id: 1 for variable_id in variable_ids},
                constant=-1,
            ),
            equality=Constraint.EQUAL_TO_ZERO,
            name=name,
            subscripts=subscripts,
        )
        for constraint_id, (name, subscripts, variable_ids) in enumerate(specs)
    ]
    if formulation == "regular":
        return constraints, None

    return constraints, ConstraintHints(
        one_hot_constraints=[
            OneHot(id=constraint_id, variables=variable_ids)
            for constraint_id, (_, _, variable_ids) in enumerate(specs)
        ]
    )


def build_knapsack_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary linear knapsack problem.

    Maximize:
        sum_i value[i] * x[i]
    Subject to:
        sum_i weight[i] * x[i] <= sum_i weight[i] // 2
        x[i] in {0, 1}
    """
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    weights = [random_generator.randint(1, 10) for _ in range(size)]
    values = [random_generator.randint(1, 20) for _ in range(size)]
    variables = [
        DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(size)
    ]

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=dict(enumerate(values))),
        constraints=[
            Constraint(
                id=0,
                function=Linear(
                    terms=dict(enumerate(weights)),
                    constant=-(sum(weights) // 2),
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="capacity",
            )
        ],
        sense=Instance.MAXIMIZE,
    )


def build_production_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build an integer linear production problem.

    Maximize:
        sum_i profit[i] * x[i]
    Subject to:
        sum_i resource[r, i] * x[i] <= 5 * sum_i resource[r, i]
            for r in {0, 1, 2}
        0 <= x[i] <= 10, x[i] is Integer
    """
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    profits = [random_generator.randint(1, 20) for _ in range(size)]
    resources = [
        [random_generator.randint(1, 10) for _ in range(size)] for _ in range(3)
    ]
    variables = [
        DecisionVariable.integer(i, lower=0, upper=10, name="x", subscripts=[i])
        for i in range(size)
    ]
    constraints = [
        Constraint(
            id=resource_id,
            function=Linear(
                terms=dict(enumerate(coefficients)),
                constant=-(sum(coefficients) * 5),
            ),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="resource",
            subscripts=[resource_id],
        )
        for resource_id, coefficients in enumerate(resources)
    ]

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=dict(enumerate(profits))),
        constraints=constraints,
        sense=Instance.MAXIMIZE,
    )


def build_blending_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a continuous linear blending problem.

    Minimize:
        sum_i cost[i] * x[i]
    Subject to:
        sum_i x[i] >= 0.25 * size
        sum_i quality[i] * x[i] >= 0.15 * size
        0 <= x[i] <= 1, x[i] is Continuous
    """
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    costs = [random_generator.uniform(0.5, 1.5) for _ in range(size)]
    qualities = [random_generator.uniform(0.5, 1.5) for _ in range(size)]
    variables = [
        DecisionVariable.continuous(i, lower=0, upper=1, name="x", subscripts=[i])
        for i in range(size)
    ]

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=dict(enumerate(costs))),
        constraints=[
            Constraint(
                id=0,
                function=Linear(
                    terms={i: -1 for i in range(size)}, constant=size * 0.25
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="demand",
            ),
            Constraint(
                id=1,
                function=Linear(
                    terms={i: -quality for i, quality in enumerate(qualities)},
                    constant=size * 0.15,
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="quality",
            ),
        ],
        sense=Instance.MINIMIZE,
    )


def build_assignment_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary linear assignment problem.

    Minimize:
        sum_{worker, task} cost[worker, task] * x[worker, task]
    Subject to:
        sum_task x[worker, task] = 1 for each worker
        sum_worker x[worker, task] = 1 for each task
        x[worker, task] in {0, 1}
    """
    _check_size(size)
    random_generator = random.Random(seed)

    def variable_id(worker: int, task: int) -> int:
        return worker * size + task

    variables = [
        DecisionVariable.binary(
            variable_id(worker, task), name="x", subscripts=[worker, task]
        )
        for worker in range(size)
        for task in range(size)
    ]
    costs = {
        variable_id(worker, task): random_generator.uniform(0.5, 1.5)
        for worker in range(size)
        for task in range(size)
    }
    specs = [
        (
            "one-task",
            [worker],
            [variable_id(worker, task) for task in range(size)],
        )
        for worker in range(size)
    ] + [
        (
            "one-worker",
            [task],
            [variable_id(worker, task) for worker in range(size)],
        )
        for task in range(size)
    ]
    constraints, constraint_hints = _build_one_hot_constraints(specs, formulation)

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=costs),
        constraints=constraints,
        sense=Instance.MINIMIZE,
        constraint_hints=constraint_hints,
    )


def build_one_hot_preparation_instance(
    size: int,
    seed: int = 0,
    formulation: str = "one-hot",
) -> Instance:
    """Build the OMMX v2 baseline for the preparation benchmark workload."""
    _check_size(size, minimum=2)
    if formulation != "one-hot":
        raise ValueError(
            "The one-hot-preparation Instance only supports one-hot formulation"
        )
    random_generator = random.Random(seed)

    def variable_id(group: int, choice: int) -> int:
        return group * size + choice

    variables = [
        DecisionVariable.binary(
            variable_id(group, choice),
            name="x",
            subscripts=[group, choice],
        )
        for group in range(size)
        for choice in range(size)
    ]
    objective = Linear(
        terms={
            variable.id: random_generator.uniform(0.5, 1.5) for variable in variables
        }
    )
    specs = [
        (
            "one-choice",
            [group],
            [variable_id(group, choice) for choice in range(size)],
        )
        for group in range(size)
    ]
    constraints, constraint_hints = _build_one_hot_constraints(specs, formulation)

    return Instance.from_components(
        decision_variables=variables,
        objective=objective,
        constraints=constraints,
        sense=Instance.MINIMIZE,
        constraint_hints=constraint_hints,
    )


def build_facility_location_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a mixed linear facility-location problem.

    Minimize:
        sum_f open_cost[f] * open[f]
        + sum_{c, f} assign_cost[c, f] * assign[c, f]
    Subject to:
        sum_f assign[c, f] <= 1 for each customer c
        assign[c, f] <= open[f] for each customer c and facility f
        open[f] in {0, 1}
        0 <= assign[c, f] <= 1, assign[c, f] is Continuous
    """
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)

    def assignment_id(customer: int, facility: int) -> int:
        return size + customer * size + facility

    variables = [
        DecisionVariable.binary(facility, name="open", subscripts=[facility])
        for facility in range(size)
    ] + [
        DecisionVariable.continuous(
            assignment_id(customer, facility),
            lower=0,
            upper=1,
            name="assign",
            subscripts=[customer, facility],
        )
        for customer in range(size)
        for facility in range(size)
    ]
    objective_terms = {
        facility: random_generator.uniform(1, 2) for facility in range(size)
    }
    objective_terms.update(
        {
            assignment_id(customer, facility): random_generator.uniform(0.5, 1.5)
            for customer in range(size)
            for facility in range(size)
        }
    )
    constraints = [
        Constraint(
            id=customer,
            function=Linear(
                terms={
                    assignment_id(customer, facility): 1 for facility in range(size)
                },
                constant=-1,
            ),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="assignment-limit",
            subscripts=[customer],
        )
        for customer in range(size)
    ]
    constraints.extend(
        Constraint(
            id=size + customer * size + facility,
            function=Linear(terms={assignment_id(customer, facility): 1, facility: -1}),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="open-facility",
            subscripts=[customer, facility],
        )
        for customer in range(size)
        for facility in range(size)
    )

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=objective_terms),
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )


def _build_portfolio_objective(size: int, random_generator: random.Random) -> Quadratic:
    factors = [random_generator.uniform(0.1, 0.5) for _ in range(size)]
    returns = [random_generator.uniform(0.05, 0.2) for _ in range(size)]
    columns = []
    rows = []
    values = []
    for i in range(size):
        for j in range(size):
            columns.append(i)
            rows.append(j)
            values.append(factors[i] * factors[j] + (0.1 if i == j else 0))

    return Quadratic(
        columns=columns,
        rows=rows,
        values=values,
        linear=Linear(terms={i: -value for i, value in enumerate(returns)}),
    )


def build_portfolio_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a continuous quadratic portfolio problem.

    Minimize:
        sum_{i, j} risk[i, j] * x[i] * x[j] - sum_i return[i] * x[i]
    Subject to:
        sum_i x[i] <= 1
        0 <= x[i] <= 1, x[i] is Continuous
    """
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    variables = [
        DecisionVariable.continuous(i, lower=0, upper=1, name="x", subscripts=[i])
        for i in range(size)
    ]

    return Instance.from_components(
        decision_variables=variables,
        objective=_build_portfolio_objective(size, random_generator),
        constraints=[
            Constraint(
                id=0,
                function=Linear(terms={i: 1 for i in range(size)}, constant=-1),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="budget",
            )
        ],
        sense=Instance.MINIMIZE,
    )


def build_portfolio_cardinality_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a mixed quadratic portfolio with a cardinality constraint.

    Minimize:
        sum_{i, j} risk[i, j] * x[i] * x[j] - sum_i return[i] * x[i]
    Subject to:
        sum_i x[i] <= 1
        x[i] <= z[i] for each asset i
        sum_i z[i] <= max(1, size // 4)
        0 <= x[i] <= 1, x[i] is Continuous
        z[i] in {0, 1}
    """
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    cardinality = max(1, size // 4)
    variables = [
        DecisionVariable.continuous(i, lower=0, upper=1, name="x", subscripts=[i])
        for i in range(size)
    ] + [
        DecisionVariable.binary(size + i, name="z", subscripts=[i]) for i in range(size)
    ]
    constraints = [
        Constraint(
            id=0,
            function=Linear(terms={i: 1 for i in range(size)}, constant=-1),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="budget",
        )
    ]
    constraints.extend(
        Constraint(
            id=i + 1,
            function=Linear(terms={i: 1, size + i: -1}),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="selection",
            subscripts=[i],
        )
        for i in range(size)
    )
    constraints.append(
        Constraint(
            id=size + 1,
            function=Linear(
                terms={size + i: 1 for i in range(size)},
                constant=-cardinality,
            ),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="cardinality",
        )
    )

    return Instance.from_components(
        decision_variables=variables,
        objective=_build_portfolio_objective(size, random_generator),
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )


def build_unit_commitment_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a mixed quadratic unit-commitment problem.

    Minimize:
        sum_i (
            quadratic_cost[i] * p[i]^2
            + production_cost[i] * p[i]
            + startup_cost[i] * u[i]
        )
    Subject to:
        p[i] <= 10 * u[i] for each generator i
        sum_i p[i] >= 5 * size
        u[i] in {0, 1}
        0 <= p[i] <= 10, p[i] is Integer
    """
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    quadratic_costs = [random_generator.uniform(0.1, 0.5) for _ in range(size)]
    production_costs = [random_generator.uniform(0.5, 1.5) for _ in range(size)]
    startup_costs = [random_generator.uniform(1, 2) for _ in range(size)]

    def production_id(generator: int) -> int:
        return size + generator

    variables = [
        DecisionVariable.binary(i, name="u", subscripts=[i]) for i in range(size)
    ] + [
        DecisionVariable.integer(
            production_id(i), lower=0, upper=10, name="p", subscripts=[i]
        )
        for i in range(size)
    ]
    objective = Quadratic(
        columns=[production_id(i) for i in range(size)],
        rows=[production_id(i) for i in range(size)],
        values=quadratic_costs,
        linear=Linear(
            terms={
                **{production_id(i): production_costs[i] for i in range(size)},
                **{i: startup_costs[i] for i in range(size)},
            }
        ),
    )
    constraints = [
        Constraint(
            id=i,
            function=Linear(terms={production_id(i): 1, i: -10}),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="production-limit",
            subscripts=[i],
        )
        for i in range(size)
    ]
    constraints.append(
        Constraint(
            id=size,
            function=Linear(
                terms={production_id(i): -1 for i in range(size)},
                constant=size * 5,
            ),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="demand",
        )
    )

    return Instance.from_components(
        decision_variables=variables,
        objective=objective,
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )


def build_clique_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary clique feasibility problem with a quadratic constraint.

    Minimize:
        0
    Subject to:
        sum_v x[v] = K
        sum_{(u, v) in edges} x[u] * x[v] = K * (K - 1) / 2
        K = (size + 1) // 2
        x[v] in {0, 1}
    """
    _check_size(size, minimum=2)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    clique_size = (size + 1) // 2
    random_vertex_count = size - clique_size
    clique_vertices = range(random_vertex_count, size)

    edges = [
        (u, v)
        for u in range(random_vertex_count)
        for v in range(u + 1, random_vertex_count)
        if random_generator.random() < 0.2
    ]
    edges.extend((u, v) for u in clique_vertices for v in range(u + 1, size))
    edges.extend((u, random_vertex_count) for u in range(random_vertex_count))

    variables = [
        DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(size)
    ]
    constraints = [
        Constraint(
            id=0,
            function=Linear(
                terms={i: 1 for i in range(size)},
                constant=-clique_size,
            ),
            equality=Constraint.EQUAL_TO_ZERO,
            name="clique-size",
        ),
        Constraint(
            id=1,
            function=Quadratic(
                columns=[u for u, _ in edges],
                rows=[v for _, v in edges],
                values=[1 for _ in edges],
                linear=Linear(
                    terms={},
                    constant=-(clique_size * (clique_size - 1) // 2),
                ),
            ),
            equality=Constraint.EQUAL_TO_ZERO,
            name="complete-subgraph",
        ),
    ]

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms={}),
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )


def build_tsp_instance(
    num_cities: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary quadratic traveling-salesperson problem.

    Minimize:
        sum_{t, i, j} distance[i, j] * x[i, t] * x[j, next(t)]
    Subject to:
        sum_i x[i, t] = 1 for each time t
        sum_t x[i, t] = 1 for each city i
        next(t) = (t + 1) % num_cities
        x[i, t] in {0, 1}
    """
    _check_size(num_cities, minimum=2)

    random_generator = random.Random(seed)
    coordinates = [
        (random_generator.random(), random_generator.random())
        for _ in range(num_cities)
    ]
    distances = [
        [math.dist(coordinates[i], coordinates[j]) for j in range(num_cities)]
        for i in range(num_cities)
    ]
    max_distance = max(max(row) for row in distances)
    distances = [[distance / max_distance for distance in row] for row in distances]

    def variable_id(city: int, time: int) -> int:
        return city * num_cities + time

    decision_variables = [
        DecisionVariable.binary(
            variable_id(city, time), name="x", subscripts=[city, time]
        )
        for city in range(num_cities)
        for time in range(num_cities)
    ]

    columns = []
    rows = []
    values = []
    for time in range(num_cities):
        next_time = (time + 1) % num_cities
        for city_i in range(num_cities):
            for city_j in range(num_cities):
                distance = distances[city_i][city_j]
                if distance == 0:
                    continue
                columns.append(variable_id(city_i, time))
                rows.append(variable_id(city_j, next_time))
                values.append(distance)

    objective = Quadratic(
        columns=columns,
        rows=rows,
        values=values,
    )

    constraint_specs = [
        (
            "one-city",
            [time],
            [variable_id(city, time) for city in range(num_cities)],
        )
        for time in range(num_cities)
    ] + [
        (
            "one-time",
            [city],
            [variable_id(city, time) for time in range(num_cities)],
        )
        for city in range(num_cities)
    ]
    constraints, constraint_hints = _build_one_hot_constraints(
        constraint_specs, formulation
    )

    return Instance.from_components(
        decision_variables=decision_variables,
        objective=objective,
        constraints=constraints,
        sense=Instance.MINIMIZE,
        constraint_hints=constraint_hints,
    )
