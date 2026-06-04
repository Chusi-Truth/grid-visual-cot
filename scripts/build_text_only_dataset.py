#!/usr/bin/env python3
"""Convert a state-token dataset into a strict text-only baseline dataset."""

import argparse
import json
import re
from pathlib import Path


CHECK_LINE_PATTERN = re.compile(
    r"^\((?:Visual check|Checking the grid|Grid snapshot):.*\)$"
)


def strip_grid_state_markup(text: str) -> str:
    lines = text.splitlines()
    kept = []
    skip_next_check = False

    for line in lines:
        stripped = line.strip()
        if stripped == "<grid_token>":
            skip_next_check = True
            continue
        if skip_next_check and CHECK_LINE_PATTERN.match(stripped):
            skip_next_check = False
            continue
        skip_next_check = False
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


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
                    "value": strip_grid_state_markup(item["conversations"][1]["value"]),
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
