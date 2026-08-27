import argparse
import gc
import statistics
import time

from common import (
    FORMULATIONS,
    INSTANCE_NAMES,
    OPERATIONS,
    PACKAGE_VERSIONS,
    SPECIAL_CONSTRAINT_CASES,
    build_instance,
    make_benchmark_operation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=OPERATIONS)
    parser.add_argument("--instance", choices=INSTANCE_NAMES, default="tsp")
    parser.add_argument("--formulation", choices=FORMULATIONS, default="regular")
    parser.add_argument(
        "--special-constraints", choices=SPECIAL_CONSTRAINT_CASES, default="none"
    )
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--solver-time-limit-ms", type=int, default=10_000)
    args = parser.parse_args()
    default_warmup = 0 if args.operation == "end-to-end" else 3
    default_repeat = 3 if args.operation == "end-to-end" else 20
    warmup = args.warmup if args.warmup is not None else default_warmup
    repeat = args.repeat if args.repeat is not None else default_repeat
    if warmup < 0:
        parser.error("--warmup must be non-negative")
    if repeat < 1:
        parser.error("--repeat must be positive")

    try:
        instance = build_instance(
            args.instance,
            args.size,
            args.seed,
            args.formulation,
            args.special_constraints,
        )
    except ValueError as error:
        parser.error(str(error))
    benchmark = make_benchmark_operation(
        args.operation,
        instance,
        args.solver_time_limit_ms,
    )

    print(
        "operation,instance,formulation,special_constraints,preparation,size,"
        "warmup,repeat,first_seconds,median_seconds,ommx_version,amplify_version,"
        "adapter_version"
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

    for _ in range(warmup):
        context = benchmark.setup()
        result = benchmark.run(context)
        del result
        del context

    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
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
        "none",
        args.size,
        warmup,
        repeat,
        f"{first_seconds:.9f}",
        f"{statistics.median(samples):.9f}",
        *PACKAGE_VERSIONS,
        sep=",",
    )


if __name__ == "__main__":
    main()
