#!/usr/bin/env python3
import argparse
import sys

from oled_status_display import OledStatusDisplay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2c-dev", default="/dev/i2c-7")
    parser.add_argument("--i2c-addr", type=lambda value: int(value, 0), default=0x3C)
    parser.add_argument("--text", default="hello world")
    args = parser.parse_args()

    try:
        display = OledStatusDisplay(
            enabled=True,
            i2c_device=args.i2c_dev,
            i2c_address=args.i2c_addr,
        )
        display.write_text(args.text)
        print(f"[OLED] wrote '{args.text}' to {args.i2c_dev} addr=0x{args.i2c_addr:02x}")
        return 0
    except Exception as exc:
        print(f"[OLED] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
