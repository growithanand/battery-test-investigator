"""Source-verified battery metadata lookup; no embeddings or generation yet."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/manifests/protocol-sources.json"
RETRIEVER_VERSION = "battery-metadata-v1"
B0005_PROTOCOL_SHA256 = "2b0614ccfdde3ca9c0466ad3e0c03903a1dbe98aec22dbe7ea410219cde51f90"
B0005_CAPACITY_CRITERION = "a 30% fade in rated capacity (from 2Ahr to 1.4Ahr)"
B0005_IMPEDANCE_METHOD = "Impedance measurement was carried out through an electrochemical impedance spectroscopy (EIS) frequency sweep"


def _verified_b0005_evidence_id(protocol: dict, excerpt: str) -> str | None:
    if protocol["battery_id"] != "B0005" or protocol["status"] not in ("available", "partial"):
        return None
    for passage in protocol["evidence"]:
        if (passage.get("source_sha256") == B0005_PROTOCOL_SHA256
                and "B0005" in passage["applicable_batteries"]
                and excerpt in passage["text"]):
            return passage["evidence_id"]
    return None


def documented_capacity_endpoint(protocol: dict) -> dict | None:
    """Expose the reviewed criterion only with its verified B0005 source passage."""
    evidence_id = _verified_b0005_evidence_id(protocol, B0005_CAPACITY_CRITERION)
    if evidence_id is None:
        return None
    return {"battery_id": "B0005", "rated_capacity_Ah": 2.0, "end_of_life_threshold_Ah": 1.4,
            "rated_capacity_fade_percent": 30.0, "evidence_id": evidence_id}


def documented_impedance_evidence_id(protocol: dict) -> str | None:
    """Locate the verified B0005 passage that describes the impedance measurement."""
    return _verified_b0005_evidence_id(protocol, B0005_IMPEDANCE_METHOD)


def read_catalog(catalog_path: Path) -> tuple[dict, str]:
    content = Path(catalog_path).read_bytes()
    catalog = json.loads(content.decode("utf-8"))
    if catalog["schema_version"] != 1:
        raise ValueError("Unsupported protocol catalog version")
    ids = [document["document_id"] for document in catalog["documents"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Protocol document IDs must be unique")
    return catalog, hashlib.sha256(content).hexdigest()


def protocol_source_path(root: Path, document: dict) -> Path:
    """Resolve a registered source without permitting reads/writes outside raw data."""
    path = (Path(root) / document["path"]).resolve()
    if not path.is_relative_to((Path(root) / "data/raw/nasa").resolve()):
        raise ValueError("Protocol source path must stay inside data/raw/nasa")
    return path


def verify_protocol_source(document: dict, content: bytes) -> str:
    """Check original bytes and cross-check applicability against the Files section."""
    if hashlib.sha256(content).hexdigest() != document["sha256"]:
        raise ValueError("Protocol source hash mismatch")
    if len(content) != document["size_bytes"]:
        raise ValueError("Protocol source size mismatch")
    text = content.decode("utf-8")
    files_section = text.split("Files:", 1)[1].split("Data Structure:", 1)[0] if "Files:" in text else ""
    declared = re.findall(r"^(B\d{4})\.mat\b", files_section, flags=re.MULTILINE)
    if not declared or sorted(declared) != sorted(document["applicable_batteries"]):
        raise ValueError("Protocol battery applicability does not match its Files section")
    return text


def protocol_passages(text: str) -> list[dict]:
    """Keep whole source sections and one-based inclusive line locators."""
    lines = text.splitlines()
    headings = [line.strip() for line in lines]
    passages = []
    for section, start_heading, stop_heading in (
        ("Data Description", "Data Description:", "Files:"),
        ("Discharge fields", "for discharge the fields are:", "for impedance the fields are:"),
    ):
        start = headings.index(start_heading)
        stop = headings.index(stop_heading, start + 1)
        while stop > start and not lines[stop - 1].strip():
            stop -= 1
        if stop <= start + 1:
            raise ValueError(f"Protocol section is empty: {section}")
        passages.append({
            "section": section, "start_line": start + 1, "end_line": stop,
            "text": "\n".join(lines[start:stop]),
        })
    return passages


def retrieve_protocol_evidence(
    battery_id: str,
    catalog_path: Path = CATALOG_PATH,
    *,
    root: Path = ROOT,
) -> dict:
    """Filter applicability first; missing/invalid sources never supply passages."""
    if not isinstance(battery_id, str) or not re.fullmatch(r"B\d{4}", battery_id):
        raise ValueError("battery_id must have the form B0005")
    catalog, catalog_sha256 = read_catalog(catalog_path)
    applicable = [doc for doc in catalog["documents"] if battery_id in doc["applicable_batteries"]]
    evidence, issues = [], []
    if not applicable:
        issues.append({"reason": "no_applicable_document"})
    for document in applicable:
        # Registry errors are configuration failures, not missing source evidence.
        path = protocol_source_path(root, document)
        try:
            text = verify_protocol_source(document, path.read_bytes())
            passages = protocol_passages(text)
        except FileNotFoundError:
            issues.append({"document_id": document["document_id"], "reason": "source_missing"})
            continue
        except OSError:
            issues.append({"document_id": document["document_id"], "reason": "source_unreadable"})
            continue
        except ValueError as error:
            issues.append({"document_id": document["document_id"], "reason": "source_invalid", "detail": str(error)})
            continue
        for passage in passages:
            evidence.append({
                "evidence_id": (
                    f"{document['document_id']}:{document['sha256']}:"
                    f"L{passage['start_line']}-L{passage['end_line']}"
                ),
                "document_id": document["document_id"],
                "source_kind": "experiment_readme",
                "source_sha256": document["sha256"],
                "applicable_batteries": list(document["applicable_batteries"]),
                "source_path": document["path"],
                "source_url": catalog["source_url"],
                "nested_package": document["nested_package"],
                "archive_member": document["archive_member"],
                **passage,
            })
    return {
        "status": "available" if evidence and not issues else "partial" if evidence else "unavailable",
        "battery_id": battery_id,
        "corpus_id": catalog["corpus_id"],
        "catalog_sha256": catalog_sha256,
        "retriever_version": RETRIEVER_VERSION,
        "evidence": evidence,
        "issues": issues,
        "interpretation_limit": (
            "Documentation describes intended experiment conditions, not proof that each "
            "record followed them. It does not establish pair comparability or physical cause."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery-id", required=True)
    args = parser.parse_args()
    print(json.dumps(retrieve_protocol_evidence(args.battery_id), indent=2))


if __name__ == "__main__":
    main()
