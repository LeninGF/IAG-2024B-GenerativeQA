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

If --output is omitted, the archive is created next to the input with a
timestamp: <parent>/<name>_<YYYYmmdd_HHMMSS>.zip, where <name> is the folder
name or the file stem (without extension).
"""
import argparse
import os
import zipfile
from datetime import datetime
from pathlib import Path


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


def zip_path(src: Path, out: Path) -> int:
    """Create the archive and return the number of files zipped."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out_abs = os.path.abspath(os.path.realpath(out))
    num_files = 0

    with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
        if src.is_file():
            add_file(zf, str(src), src.name)
            num_files += 1
        else:
            root_arc = src.name
            for dirpath, dirnames, filenames in os.walk(src):
                rel_dir = os.path.relpath(dirpath, src)
                dir_arc = root_arc if rel_dir == "." else os.path.join(root_arc, rel_dir)
                add_dir_entry(zf, dirpath, dir_arc)

                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    # Never include the archive being written into itself.
                    if os.path.abspath(filepath) == out_abs:
                        continue
                    rel_file = os.path.relpath(filepath, src)
                    file_arc = os.path.join(root_arc, rel_file)
                    add_file(zf, filepath, file_arc)
                    num_files += 1

    return num_files


def main():
    args = parse_args()

    src = Path(args.input).expanduser()
    if not src.exists():
        raise SystemExit(f"ERROR: input path does not exist: {src}")

    out = Path(args.output).expanduser() if args.output else default_output_path(src)
    if not str(out).lower().endswith(".zip"):
        out = Path(str(out) + ".zip")

    num_files = zip_path(src, out)
    size_bytes = out.stat().st_size
    print(f"Zipped {num_files} file(s) -> {out} ({size_bytes} bytes)")


if __name__ == "__main__":
    main()
