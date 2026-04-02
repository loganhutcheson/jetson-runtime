#!/usr/bin/env python3
import argparse
import json

from posture_classifier import (
    build_feature_windows,
    load_jsonl_reports,
    save_posture_model,
    train_posture_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", action="append", required=True, help="Path to a good-posture JSONL capture. Repeatable.")
    parser.add_argument("--okay", action="append", required=True, help="Path to an okay-posture JSONL capture. Repeatable.")
    parser.add_argument("--bad", action="append", required=True, help="Path to a bad-posture JSONL capture. Repeatable.")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--min-valid-frames", type=int, default=12)
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--report-out")
    args = parser.parse_args()

    label_to_windows = {}
    for label, paths in (("good", args.good), ("okay", args.okay), ("bad", args.bad)):
        windows = []
        for path in paths:
            windows.extend(
                build_feature_windows(
                    load_jsonl_reports(path),
                    window_size=args.window_size,
                    stride=args.window_stride,
                    min_valid_frames=args.min_valid_frames,
                )
            )
        if not windows:
            raise SystemExit(f"no training windows produced for label={label}")
        label_to_windows[label] = windows

    model, report = train_posture_model(
        label_to_windows,
        validation_stride=args.validation_stride,
        knn_k=args.knn_k,
    )
    save_posture_model(args.model_out, model)

    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    print(json.dumps(report, indent=2))
    print(f"model_out={args.model_out}")


if __name__ == "__main__":
    main()
