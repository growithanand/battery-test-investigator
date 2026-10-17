"""Fetch one source battery and its README from NASA's linked archive."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
PACKAGE = "5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip"


class RemoteZip(io.BufferedIOBase):
    """Allow ZipFile to read selected parts of NASA's archive over HTTP."""

    def __init__(self):
        with urlopen(Request(SOURCE_URL, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def read(self, count=-1):
        stop = self.size if count < 0 else min(self.size, self.position + count)
        if self.position >= stop:
            return b""
        request = Request(SOURCE_URL, headers={"Range": f"bytes={self.position}-{stop - 1}"})
        with urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("NASA's server did not honor the partial download request")
            content = response.read()
        if len(content) != stop - self.position:
            raise ValueError("Incomplete archive segment")
        self.position = stop
        return content


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    raw_dir = ROOT / "data/raw/nasa"
    raw_dir.mkdir(parents=True, exist_ok=True)
    package_path = raw_dir / "package-1.zip"
    if not package_path.exists():
        print("Fetching only the first experiment package (approximately 57 MB)...", flush=True)
        with RemoteZip() as remote, ZipFile(remote) as archive:
            package_bytes = archive.read(PACKAGE)
        package_path.write_bytes(package_bytes)
    else:
        package_bytes = package_path.read_bytes()
    package_dir = raw_dir / "package-1"
    package_dir.mkdir(exist_ok=True)
    with ZipFile(io.BytesIO(package_bytes)) as package:
        for filename in ("B0005.mat", "README.txt"):
            # Select exact known members; never extract arbitrary archive paths.
            (package_dir / filename).write_bytes(package.read(filename))

    files = []
    for filename in ("B0005.mat", "README.txt"):
        path = package_dir / filename
        files.append({"path": path.relative_to(ROOT).as_posix(), "archive_member": filename,
                      "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "source_url": SOURCE_URL,
        "nested_package": PACKAGE,
        "nested_package_sha256": hashlib.sha256(package_bytes).hexdigest(),
        "attribution": "B. Saha and K. Goebel (2007), Battery Data Set, NASA PCoE",
        "files": files,
    }
    manifest_dir = ROOT / "data/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "B0005.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Source files ready: {package_dir}")
    print(f"Source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
