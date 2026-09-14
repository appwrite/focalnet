"""Verify license metadata and files in built Python distributions."""

import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def only(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one {description}, found {len(paths)}")
    return paths[0]


def verify_wheel(path: Path, expected_license: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = only(
            [name for name in names if name.endswith(".dist-info/METADATA")],
            "wheel METADATA file",
        )
        license_name = only(
            [name for name in names if name.endswith(".dist-info/licenses/LICENSE")],
            "wheel license file",
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata["License-Expression"] != "MIT":
            raise ValueError("Wheel does not declare the MIT license expression")
        if "LICENSE" not in metadata.get_all("License-File", []):
            raise ValueError("Wheel metadata does not reference LICENSE")
        if archive.read(license_name) != expected_license:
            raise ValueError("Wheel license differs from the repository LICENSE")


def verify_sdist(path: Path, expected_license: bytes) -> None:
    with tarfile.open(path, "r:gz") as archive:
        member = only(
            [member for member in archive.getmembers() if member.name.endswith("/LICENSE")],
            "source-distribution license file",
        )
        handle = archive.extractfile(member)
        if handle is None or handle.read() != expected_license:
            raise ValueError("Source-distribution license differs from the repository LICENSE")


def main() -> None:
    distribution = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    expected_license = Path("LICENSE").read_bytes()
    verify_wheel(only(list(distribution.glob("*.whl")), "wheel"), expected_license)
    verify_sdist(only(list(distribution.glob("*.tar.gz")), "source distribution"), expected_license)
    print("Package license verification passed")


if __name__ == "__main__":
    main()
