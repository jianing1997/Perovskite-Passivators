from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the separately archived data/model release."
    )
    parser.add_argument("release_dir", type=Path)
    args = parser.parse_args()

    release_dir = args.release_dir.resolve()
    if not release_dir.is_dir():
        raise NotADirectoryError(release_dir)

    runtime_assets = release_dir / "runtime_assets"
    if runtime_assets.exists():
        shutil.copytree(runtime_assets, REPO_ROOT, dirs_exist_ok=True)

    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    for folder in ("results", "source_data"):
        source = release_dir / folder
        if source.exists():
            shutil.copytree(source, data_dir / folder, dirs_exist_ok=True)

    print(f"Installed external release from {release_dir}")


if __name__ == "__main__":
    main()
