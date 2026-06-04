#!/usr/bin/env python3
"""Convert a grid-token dataset into a textual-state-description ablation dataset."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data import build_textual_state_desc_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    with src.open() as f:
        data = json.load(f)

    out = []
    for item in data:
        new_item = {
            "images": [item["images"][0]],
            "conversations": [
                item["conversations"][0],
                {
                    "from": item["conversations"][1]["from"],
                    "value": build_textual_state_desc_response(item["conversations"][1]["value"]),
                },
            ],
        }
        out.append(new_item)

    with dst.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved {dst}")
    print(f"samples {len(out)}")


if __name__ == "__main__":
    main()
