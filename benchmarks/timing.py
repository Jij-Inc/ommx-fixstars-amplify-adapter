import argparse
import gc
import statistics
import time

from common import (
    FORMULATIONS,
    INSTANCE_NAMES,
    PACKAGE_VERSIONS,
    PREPARATIONS,
    SPECIAL_CONSTRAINT_CASES,
    build_instance,
    make_benchmark_operation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation", choices=("prepare", "instance-to-model", "result-to-solution")
    )
    parser.add_argument("--instance", choices=INSTANCE_NAMES, default="tsp")
    parser.add_argument("--formulation", choices=FORMULATIONS, default="regular")
    parser.add_argument(
        "--special-constraints", choices=SPECIAL_CONSTRAINT_CASES, default="none"
    )
    parser.add_argument("--preparation", choices=PREPARATIONS, default="none")
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--solver-time-limit-ms", type=int, default=10_000)
    args = parser.parse_args()

    try:
        instance = build_instance(
            args.instance,
            args.size,
            args.seed,
            args.formulation,
            args.special_constraints,
            args.preparation,
        )
        benchmark = make_benchmark_operation(
            args.operation,
            instance,
            args.solver_time_limit_ms,
            args.special_constraints,
            args.preparation,
        )
    except ValueError as error:
        parser.error(str(error))

    print(
        "operation,instance,formulation,special_constraints,preparation,size,"
        "first_seconds,median_seconds,ommx_version,amplify_version,adapter_version"
    )

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        gc.collect()
        first_context = benchmark.setup()
        start = time.perf_counter()
        first_result = benchmark.run(first_context)
        first_seconds = time.perf_counter() - start
        del first_result
        del first_context
    finally:
        if gc_was_enabled:
            gc.enable()

    for _ in range(args.warmup):
        context = benchmark.setup()
        result = benchmark.run(context)
        del result
        del context

    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(args.repeat):
            gc.collect()
            context = benchmark.setup()
            start = time.perf_counter()
            result = benchmark.run(context)
            elapsed = time.perf_counter() - start
            del result
            del context
            samples.append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    print(
        args.operation,
        args.instance,
        args.formulation,
        args.special_constraints,
        args.preparation,
        args.size,
        f"{first_seconds:.9f}",
        f"{statistics.median(samples):.9f}",
        *PACKAGE_VERSIONS,
        sep=",",
    )


if __name__ == "__main__":
    main()
