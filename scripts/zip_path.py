"""Zip a single file or folder into a .zip archive.

Useful for packaging produced datasets, worker logs, or any output folder
before moving/uploading it. Originals are never modified.

Usage:
    # Zip a folder, keeping the folder name as the top-level archive entry:
    python scripts/zip_path.py --input dataset

    # Zip a single file:
    python scripts/zip_path.py --input dataset/dataset_squad.json

    # Write to an explicit destination (parent dirs are created as needed):
    python scripts/zip_path.py --input logs --output /tmp/logs_backup.zip

    # Skip Hugging Face model directories (model/, checkpoints/) while zipping
    # a results/experiment folder (off by default):
    python scripts/zip_path.py --input out_experiments/run1 --skip-models

If --output is omitted, the archive is created next to the input with a
timestamp: <parent>/<name>_<YYYYmmdd_HHMMSS>.zip, where <name> is the folder
name or the file stem (without extension).

`--skip-models` excludes every directory named `model` or `checkpoints` (and
everything beneath them) from folder archives --- this avoids packaging large
Hugging Face checkpoints while keeping results such as .csv/.jsonl/.json/.pdf/
.png files.

For folder inputs a bytes-based progress bar is shown (elapsed time, MB, ETA)
via tqdm if available; otherwise a stdlib-only counting fallback is used.
"""
import argparse
import os
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm may be absent on stdlib-only envs
    tqdm = None

# Directory names treated as "models" and skipped when --skip-models is set.
SKIP_MODEL_DIRS = {"model", "checkpoints"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zip a single file or folder into a .zip archive (originals kept)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="File or folder to zip (required)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination .zip path (default: <parent>/<name>_<timestamp>.zip "
             "next to the input)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="When zipping a folder, exclude directories named 'model' or "
             "'checkpoints' (and their contents). No effect on single files.",
    )
    return parser.parse_args()


def default_output_path(src: Path) -> Path:
    """Return a timestamped .zip path next to the source path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = src.name if src.is_dir() else src.stem
    return src.parent / f"{name}_{timestamp}.zip"


def normalize_arcname(path: str) -> str:
    """Convert an OS path to a forward-slash archive entry name."""
    return path.replace(os.sep, "/")


def add_dir_entry(zf: zipfile.ZipFile, path: str, arcname: str) -> None:
    """Add a directory entry to the archive (name ends with '/')."""
    zf.write(path, normalize_arcname(arcname) + "/")


def add_file(zf: zipfile.ZipFile, path: str, arcname: str) -> None:
    """Add a file to the archive."""
    zf.write(path, normalize_arcname(arcname))


def collect_files(src: Path, skip_models: bool):
    """Return (files, total_bytes) for a folder walk, honoring skip_models.

    `files` is a list of (filepath, root_arc, rel_file) tuples ready to be
    added to the archive. Directory entries are handled separately during the
    walk; this only collects file paths and their byte sizes.
    """
    files = []
    total_bytes = 0
    root_arc = src.name
    for dirpath, dirnames, filenames in os.walk(src):
        if skip_models:
            # Prune model/checkpoints dirs so neither they nor their contents
            # are visited or archived.
            dirnames[:] = [d for d in dirnames if d not in SKIP_MODEL_DIRS]

        rel_dir = os.path.relpath(dirpath, src)
        dir_arc = root_arc if rel_dir == "." else os.path.join(root_arc, rel_dir)
        files.append((dirpath, dir_arc, None))  # directory entry sentinel

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            rel_file = os.path.relpath(filepath, src)
            file_arc = os.path.join(root_arc, rel_file)
            files.append((filepath, file_arc, filename))
            try:
                total_bytes += os.path.getsize(filepath)
            except OSError:
                pass
    return files, total_bytes


def zip_path(src: Path, out: Path, skip_models: bool = False) -> int:
    """Create the archive and return the number of files zipped."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out_abs = os.path.abspath(os.path.realpath(out))
    num_files = 0

    with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
        if src.is_file():
            add_file(zf, str(src), src.name)
            num_files += 1
        else:
            files, total_bytes = collect_files(src, skip_models)

            if tqdm is not None:
                bar = tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc="Zipping",
                    mininterval=0.2,
                )
                for filepath, file_arc, filename in files:
                    if filename is None:
                        # Directory entry.
                        add_dir_entry(zf, filepath, file_arc)
                        continue
                    if os.path.abspath(filepath) == out_abs:
                        continue  # never include the archive being written
                    size = os.path.getsize(filepath)
                    add_file(zf, filepath, file_arc)
                    num_files += 1
                    bar.update(size)
                bar.close()
            else:
                # Stdlib-only fallback: sparse progress lines by count.
                n_total = sum(1 for f, _, fn in files if fn is not None)
                n_done = 0
                processed = 0
                for filepath, file_arc, filename in files:
                    if filename is None:
                        add_dir_entry(zf, filepath, file_arc)
                        continue
                    if os.path.abspath(filepath) == out_abs:
                        continue
                    size = os.path.getsize(filepath)
                    add_file(zf, filepath, file_arc)
                    num_files += 1
                    n_done += 1
                    processed += size
                    if n_done % 50 == 0 or n_done == n_total:
                        pct = (processed / total_bytes * 100) if total_bytes else 0.0
                        print(f"  zipped {n_done}/{n_total} files "
                              f"({processed} bytes, {pct:.1f}%)")

    return num_files


def main():
    args = parse_args()

    src = Path(args.input).expanduser()
    if not src.exists():
        raise SystemExit(f"ERROR: input path does not exist: {src}")

    out = Path(args.output).expanduser() if args.output else default_output_path(src)
    if not str(out).lower().endswith(".zip"):
        out = Path(str(out) + ".zip")

    num_files = zip_path(src, out, skip_models=args.skip_models)
    size_bytes = out.stat().st_size
    print(f"\nZipped {num_files} file(s) -> {out} ({size_bytes} bytes)")


if __name__ == "__main__":
    main()
