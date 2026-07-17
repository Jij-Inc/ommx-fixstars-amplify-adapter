import math
import random

from ommx import (
    Constraint,
    DecisionVariable,
    Instance,
    Linear,
    OneHotConstraint,
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
) -> tuple[dict[int, Constraint], dict[int, OneHotConstraint]]:
    if formulation not in ("regular", "one-hot"):
        raise ValueError(f"Unknown formulation: {formulation}")

    if formulation == "regular":
        return (
            {
                constraint_id: Constraint(
                    function=Linear(
                        terms={variable_id: 1 for variable_id in variable_ids},
                        constant=-1,
                    ),
                    equality=Constraint.EQUAL_TO_ZERO,
                    name=name,
                    subscripts=subscripts,
                )
                for constraint_id, (name, subscripts, variable_ids) in enumerate(specs)
            },
            {},
        )

    return {}, {
        constraint_id: OneHotConstraint(
            variables=variable_ids,
            name=name,
            subscripts=subscripts,
        )
        for constraint_id, (name, subscripts, variable_ids) in enumerate(specs)
    }


def build_knapsack_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary linear problem to measure Binary and linear conversion."""
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
        constraints={
            0: Constraint(
                function=Linear(
                    terms=dict(enumerate(weights)),
                    constant=-(sum(weights) // 2),
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="capacity",
            )
        },
        sense=Instance.MAXIMIZE,
    )


def build_production_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build an integer linear problem to measure bounded Integer conversion."""
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
    constraints = {
        resource_id: Constraint(
            function=Linear(
                terms=dict(enumerate(coefficients)),
                constant=-(sum(coefficients) * 5),
            ),
            equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            name="resource",
            subscripts=[resource_id],
        )
        for resource_id, coefficients in enumerate(resources)
    }

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=dict(enumerate(profits))),
        constraints=constraints,
        sense=Instance.MAXIMIZE,
    )


def build_blending_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a continuous linear problem to measure Real and bound conversion."""
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
        constraints={
            0: Constraint(
                function=Linear(
                    terms={i: -1 for i in range(size)}, constant=size * 0.25
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="demand",
            ),
            1: Constraint(
                function=Linear(
                    terms={i: -quality for i, quality in enumerate(qualities)},
                    constant=size * 0.15,
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="quality",
            ),
        },
        sense=Instance.MINIMIZE,
    )


def build_assignment_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary linear problem to compare regular and OneHot constraints."""
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
    constraints, one_hot_constraints = _build_one_hot_constraints(specs, formulation)

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=costs),
        constraints=constraints,
        one_hot_constraints=one_hot_constraints,
        sense=Instance.MINIMIZE,
    )


def build_facility_location_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a mixed linear problem to measure Binary and Continuous conversion."""
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
    constraints = {
        customer: Constraint(
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
    }
    constraints.update(
        {
            size + customer * size + facility: Constraint(
                function=Linear(
                    terms={assignment_id(customer, facility): 1, facility: -1}
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="open-facility",
                subscripts=[customer, facility],
            )
            for customer in range(size)
            for facility in range(size)
        }
    )

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=objective_terms),
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )


def build_portfolio_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a continuous quadratic problem to measure Real quadratic conversion."""
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    factors = [random_generator.uniform(0.1, 0.5) for _ in range(size)]
    returns = [random_generator.uniform(0.05, 0.2) for _ in range(size)]
    variables = [
        DecisionVariable.continuous(i, lower=0, upper=1, name="x", subscripts=[i])
        for i in range(size)
    ]
    columns = []
    rows = []
    values = []
    for i in range(size):
        for j in range(size):
            columns.append(i)
            rows.append(j)
            values.append(factors[i] * factors[j] + (0.1 if i == j else 0))

    return Instance.from_components(
        decision_variables=variables,
        objective=Quadratic(
            columns=columns,
            rows=rows,
            values=values,
            linear=Linear(terms={i: -value for i, value in enumerate(returns)}),
        ),
        constraints={
            0: Constraint(
                function=Linear(terms={i: 1 for i in range(size)}, constant=-1),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="budget",
            )
        },
        sense=Instance.MINIMIZE,
    )


def build_clique_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary feasibility problem to measure quadratic constraints."""
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
    constraints = {
        0: Constraint(
            function=Linear(
                terms={i: 1 for i in range(size)},
                constant=-clique_size,
            ),
            equality=Constraint.EQUAL_TO_ZERO,
            name="clique-size",
        ),
        1: Constraint(
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
    }

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms={}),
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )


def build_tsp_instance(
    num_cities: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary quadratic TSP to compare regular and OneHot constraints."""
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
    constraints, one_hot_constraints = _build_one_hot_constraints(
        constraint_specs, formulation
    )

    return Instance.from_components(
        decision_variables=decision_variables,
        objective=objective,
        constraints=constraints,
        one_hot_constraints=one_hot_constraints,
        sense=Instance.MINIMIZE,
    )
