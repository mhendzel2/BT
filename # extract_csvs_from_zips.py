# extract_csvs_from_zips.py
#
# Scans a directory for .zip files. For each zip, finds the single CSV inside
# (case-insensitive) and writes it into the zip's parent directory.
#
# Default behavior: output directory = the same folder that contains the .zip files.
# If you truly mean "one level up from the folder of zips", run with: --out "<zip_dir>\.."

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def iter_zip_files(root: Path, recursive: bool) -> list[Path]:
    paths: list[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".zip":
                paths.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() == ".zip":
                paths.append(p)
    return sorted(paths)


def safe_unique_path(out_dir: Path, desired_name: str, zip_stem: str, overwrite: bool) -> Path:
    dest = out_dir / desired_name
    if overwrite or not dest.exists():
        return dest

    # Avoid overwriting: prefix with zip name; if still collides, add counter.
    base = f"{zip_stem}__{desired_name}"
    candidate = out_dir / base
    if not candidate.exists():
        return candidate

    stem = Path(base).stem
    suffix = Path(base).suffix
    for i in range(2, 10_000):
        candidate = out_dir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find a unique filename for {desired_name} in {out_dir}")


def extract_single_csv(zip_path: Path, out_dir: Path, overwrite: bool) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [
            info for info in zf.infolist()
            if (not info.is_dir()) and info.filename.lower().endswith(".csv")
        ]

        if len(csv_members) != 1:
            raise ValueError(
                f"Expected exactly 1 CSV in {zip_path.name}, found {len(csv_members)}: "
                f"{[m.filename for m in csv_members]}"
            )

        member = csv_members[0]
        csv_basename = Path(member.filename).name  # flatten any internal folders
        dest_path = safe_unique_path(out_dir, csv_basename, zip_path.stem, overwrite)

        # Stream copy to avoid zip-slip issues and to handle large files efficiently.
        with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        return dest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the single CSV from each ZIP file in a folder and place it in the ZIP's parent directory."
    )
    parser.add_argument(
        "zip_dir",
        type=Path,
        nargs="?",
        default=Path(r"C:\Users\mjhen\Github\UWdata"),
        help=r'Directory containing .zip files (default: C:\Users\mjhen\Github\UWdata)',
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for extracted CSVs (default: same as zip_dir).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search subfolders for .zip files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files instead of renaming to avoid collisions.",
    )

    args = parser.parse_args()
    zip_dir: Path = args.zip_dir
    out_dir: Path = args.out if args.out is not None else zip_dir

    if not zip_dir.exists() or not zip_dir.is_dir():
        print(f"ERROR: zip_dir does not exist or is not a directory: {zip_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    zip_files = iter_zip_files(zip_dir, args.recursive)
    if not zip_files:
        print(f"No .zip files found in: {zip_dir}")
        return 0

    ok = 0
    failed = 0

    for zp in zip_files:
        try:
            dest = extract_single_csv(zp, out_dir, args.overwrite)
            print(f"[OK] {zp.name} -> {dest.name}")
            ok += 1
        except zipfile.BadZipFile:
            print(f"[FAIL] {zp.name}: not a valid zip file", file=sys.stderr)
            failed += 1
        except (RuntimeError, ValueError) as e:
            print(f"[FAIL] {zp.name}: {e}", file=sys.stderr)
            failed += 1
        except Exception as e:
            print(f"[FAIL] {zp.name}: unexpected error: {e}", file=sys.stderr)
            failed += 1

    print(f"Done. Extracted: {ok}, Failed: {failed}, Output dir: {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
