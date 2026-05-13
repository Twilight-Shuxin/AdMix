"""
Delete all files associated with a given trace tag across data root folders.

Usage:
    python delete_trace_files.py <trace_tag> --model MODEL [--baseline BASELINE] [--data-roots ROOT1 ROOT2 ...] [--dry-run]

Example:
    python delete_trace_files.py "Azure1,2000,100,3600,5,0.05" --model qwen14b --dry-run
    python delete_trace_files.py "Azure1,2000,100,3600,5,0.05" --model qwen14b --baseline utils
    python delete_trace_files.py "Azure1,2000,100,3600,5,0.05" --model llama7b --data-roots data_final
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent


def find_trace_files(data_root: Path, trace_tag: str, model: str, baseline: Optional[str]) -> list[Path]:
    """Find all files under data_root/{model}/{baseline}/ containing the trace tag."""
    tag_pattern = f"[{trace_tag}]"
    matched = []

    model_dir = data_root / model
    if not model_dir.is_dir():
        print(f"  Warning: {model_dir} does not exist, skipping.", file=sys.stderr)
        return matched

    baseline_dirs = [model_dir / baseline] if baseline else sorted(
        d for d in model_dir.iterdir() if d.is_dir()
    )

    for baseline_dir in baseline_dirs:
        if not baseline_dir.is_dir():
            print(f"  Warning: {baseline_dir} does not exist, skipping.", file=sys.stderr)
            continue
        for f in baseline_dir.rglob("*"):
            if f.is_file() and tag_pattern in f.name:
                matched.append(f)

    return matched


def main():
    parser = argparse.ArgumentParser(description="Delete files for a given trace tag.")
    parser.add_argument("trace_tag", help='Trace tag, e.g. "Azure1,2000,100,3600,5,0.05"')
    parser.add_argument("--model", required=True, help="Model folder, e.g. qwen14b or llama7b")
    parser.add_argument("--baseline", default=None, help="Baseline subfolder, e.g. utils, mps, context (default: all)")
    parser.add_argument(
        "--data-roots",
        nargs="+",
        metavar="DIR",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be deleted without deleting them",
    )
    args = parser.parse_args()

    trace_tag = args.trace_tag.strip("[]")  # accept with or without brackets
    all_files: list[Path] = []

    for root_name in args.data_roots:
        data_root = SCRIPT_DIR / root_name
        found = find_trace_files(data_root, trace_tag, args.model, args.baseline)
        if found:
            print(f"\n{data_root} ({len(found)} files):")
            for f in found:
                print(f"  {f.relative_to(SCRIPT_DIR)}")
        all_files.extend(found)

    if not all_files:
        print(f"No files found for trace tag: [{trace_tag}]")
        return

    print(f"\nTotal: {len(all_files)} file(s) matched.")

    if args.dry_run:
        print("Dry run — no files deleted.")
        return

    confirm = input(f"\nDelete {len(all_files)} file(s)? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    deleted, failed = 0, 0
    for f in all_files:
        try:
            f.unlink()
            deleted += 1
        except OSError as e:
            print(f"  Error deleting {f}: {e}", file=sys.stderr)
            failed += 1

    print(f"Deleted {deleted} file(s)." + (f" {failed} failed." if failed else ""))


if __name__ == "__main__":
    main()
