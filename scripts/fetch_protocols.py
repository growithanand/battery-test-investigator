"""Prepare only registered protocol text, retaining source bytes outside Git."""

import argparse
from contextlib import ExitStack
import io
from pathlib import Path
from zipfile import ZipFile

from scripts.fetch_sample import RemoteZip, SOURCE_URL
from scripts.protocol_evidence import (
    CATALOG_PATH, ROOT, protocol_source_path, read_catalog, verify_protocol_source,
)


def prepare_protocol_sources(
    document_cache: Path | None = None,
    *,
    catalog_path: Path = CATALOG_PATH,
    root: Path = ROOT,
) -> list[Path]:
    """Use an explicit existing document cache, or read exact members from NASA.

    Existing destinations are verified, not overwritten. A supplied document
    cache never silently falls back to network access when a file is missing.
    """
    catalog, _ = read_catalog(catalog_path)
    if catalog["source_url"] != SOURCE_URL:
        raise ValueError("Protocol catalog must use the registered NASA archive")
    prepared = []
    with ExitStack() as stack:
        outer_archive = None
        for document in catalog["documents"]:
            destination = protocol_source_path(root, document)
            if destination.exists():
                verify_protocol_source(document, destination.read_bytes())
                prepared.append(destination)
                continue
            if document_cache is not None:
                cache_root = Path(document_cache).resolve()
                cached = (cache_root / document["cache_member"]).resolve()
                if not cached.is_relative_to(cache_root):
                    raise ValueError("Protocol cache member must stay inside the supplied cache")
                content = cached.read_bytes()
            else:
                if outer_archive is None:
                    outer_archive = stack.enter_context(ZipFile(stack.enter_context(RemoteZip())))
                package_bytes = outer_archive.read(document["nested_package"])
                with ZipFile(io.BytesIO(package_bytes)) as package:
                    content = package.read(document["archive_member"])
            verify_protocol_source(document, content)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(content)
            prepared.append(destination)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-cache", type=Path, help="Optional existing cache with registered package/member paths")
    args = parser.parse_args()
    for path in prepare_protocol_sources(args.document_cache):
        print(f"Verified protocol source: {path}")


if __name__ == "__main__":
    main()
