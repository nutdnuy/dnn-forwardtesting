from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_ohlcv
from .pipeline import ExperimentConfig, run_forwardtesting


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only DNN forward-testing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate an OHLCV CSV")
    validate.add_argument("--data", required=True, type=Path)
    run = subparsers.add_parser("run", help="run a forward-testing experiment")
    run.add_argument("--data", required=True, type=Path)
    run.add_argument("--cutoff", required=True)
    run.add_argument("--horizon", type=int, default=30)
    run.add_argument("--lookback", type=int, default=5)
    run.add_argument("--backend", choices=("last_value", "torch_mlp"), default="last_value")
    run.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    frame = load_ohlcv(args.data, min_rows=2)
    if args.command == "validate":
        print(json.dumps({"ok": True, "rows": len(frame), "columns": list(frame.columns)}))
        return 0
    config = ExperimentConfig(
        lookback=args.lookback,
        horizon=args.horizon,
        forecast_backend=args.backend,
    )
    result = run_forwardtesting(frame, args.cutoff, config=config, run_dir=args.output)
    print(json.dumps({
        "selected_forward": result.selected_forward.to_dict(),
        "selected_traditional": result.selected_traditional.to_dict(),
        "forward_actual": result.forward_actual.metrics,
        "traditional_actual": result.traditional_actual.metrics,
        "artifacts": result.artifacts,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
