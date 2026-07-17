import argparse
import gc
import statistics
import time

from common import (
    FORMULATIONS,
    INSTANCE_NAMES,
    PACKAGE_VERSIONS,
    build_instance,
    prepare_target,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation", choices=("instance-to-model", "result-to-solution")
    )
    parser.add_argument("--instance", choices=INSTANCE_NAMES, default="tsp")
    parser.add_argument("--formulation", choices=FORMULATIONS, default="regular")
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--solver-time-limit-ms", type=int, default=120_000)
    args = parser.parse_args()

    print(
        "operation,instance,formulation,size,median_seconds,ommx_version,"
        "amplify_version,adapter_version"
    )
    instance = build_instance(args.instance, args.size, args.seed, args.formulation)
    target = prepare_target(args.operation, instance, args.solver_time_limit_ms)

    for _ in range(args.warmup):
        result = target()
        del result

    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(args.repeat):
            gc.collect()
            start = time.perf_counter()
            result = target()
            elapsed = time.perf_counter() - start
            del result
            samples.append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    print(
        args.operation,
        args.instance,
        args.formulation,
        args.size,
        f"{statistics.median(samples):.9f}",
        *PACKAGE_VERSIONS,
        sep=",",
    )


if __name__ == "__main__":
    main()
