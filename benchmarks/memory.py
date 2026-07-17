import argparse
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

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
    parser.add_argument("--solver-time-limit-ms", type=int, default=10_000)
    args = parser.parse_args()

    try:
        import memray  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise SystemExit("Run with `uv run --frozen --with memray`.") from error

    instance = build_instance(args.instance, args.size, args.seed, args.formulation)
    target = prepare_target(args.operation, instance, args.solver_time_limit_ms)

    gc.collect()
    with TemporaryDirectory() as directory:
        first_capture = Path(directory) / "first.bin"
        with memray.Tracker(first_capture):
            first_result = target()
        del first_result
        first_reader = memray.FileReader(first_capture)
        first_peak_memory_bytes = first_reader.metadata.peak_memory
        first_reader.close()

        gc.collect()
        warmed_capture = Path(directory) / "warmed.bin"
        with memray.Tracker(warmed_capture):
            result = target()
        del result
        reader = memray.FileReader(warmed_capture)
        peak_memory_bytes = reader.metadata.peak_memory
        reader.close()

    print(
        "operation,instance,formulation,size,first_peak_memory_bytes,"
        "peak_memory_bytes,ommx_version,amplify_version,adapter_version"
    )
    print(
        args.operation,
        args.instance,
        args.formulation,
        args.size,
        first_peak_memory_bytes,
        peak_memory_bytes,
        *PACKAGE_VERSIONS,
        sep=",",
    )


if __name__ == "__main__":
    main()
