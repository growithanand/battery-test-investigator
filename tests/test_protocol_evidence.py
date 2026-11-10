"""Constructed text/ZIP fixtures test provenance and applicability, not NASA facts."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import fetch_protocols
from scripts.fetch_sample import SOURCE_URL
from scripts.protocol_evidence import B0005_PROTOCOL_SHA256, documented_capacity_endpoint, read_catalog, retrieve_protocol_evidence


@pytest.mark.parametrize("battery, status, source_hash, expected", [
    ("B0005", "partial", B0005_PROTOCOL_SHA256, True),
    ("B0005", "available", "0" * 64, False),
    ("B0006", "available", B0005_PROTOCOL_SHA256, False),
    ("B0005", "unavailable", B0005_PROTOCOL_SHA256, False),
])
def test_documented_capacity_criterion_requires_reviewed_b0005_source(battery, status, source_hash, expected):
    passage = {"source_sha256": source_hash, "applicable_batteries": ["B0005", "B0006"],
               "text": "Experiment ended after a 30% fade in rated capacity (from 2Ahr to 1.4Ahr).",
               "evidence_id": "test-source:L1-L2"}
    protocol = {"battery_id": battery, "status": status, "evidence": [passage]}
    endpoint = documented_capacity_endpoint(protocol)
    if expected:
        assert endpoint["rated_capacity_Ah"] == 2.0 and endpoint["end_of_life_threshold_Ah"] == 1.4
        assert endpoint["evidence_id"] == passage["evidence_id"]
    else:
        assert endpoint is None


def register_bytes(document, content):
    document.update(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


@pytest.fixture
def corpus(tmp_path):
    root, cache = tmp_path / "project", tmp_path / "cache"
    catalog = {"schema_version": 1, "corpus_id": "synthetic-v1", "source_url": SOURCE_URL, "documents": []}
    for number, battery, description in (
        (1, "B0005", "Synthetic constant 2A test; endpoint 1.4Ahr."),
        (3, "B0033", "Synthetic B0005-like capacity test; endpoint 1.6Ahr (20% fade)."),
        (5, "B0049", "Synthetic capacity test terminated by a control software crash."),
    ):
        content = (
            f"Data Description:\r\n{description}\r\n\r\nFiles:\r\n"
            f"{battery}.mat\tSynthetic battery\r\n\r\nData Structure:\r\n"
            "cycle:\r\n\tfor discharge the fields are:\r\n"
            "\t\tCapacity: supplied capacity (Ahr)\r\n"
            "\tfor impedance the fields are:\r\n\t\tRe: resistance\r\n"
        ).encode("utf-8")
        member = f"package-{number}/README.txt"
        document = {
            "document_id": f"synthetic-{battery}", "applicable_batteries": [battery],
            "path": f"data/raw/nasa/protocols/{member}",
            "nested_package": f"synthetic/package-{number}.zip",
            "archive_member": "README.txt", "cache_member": member,
        }
        register_bytes(document, content)
        for path in (root / document["path"], cache / member):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        catalog["documents"].append(document)
    catalog_path = root / "data/manifests/protocol-sources.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return root, catalog_path, cache


def test_lookup_preserves_original_passages_and_locators(corpus):
    root, catalog_path, _ = corpus
    result = retrieve_protocol_evidence("B0005", catalog_path, root=root)
    assert result["status"] == "available"
    assert result["issues"] == []
    assert len(result["evidence"]) == 2
    assert {item["document_id"] for item in result["evidence"]} == {"synthetic-B0005"}
    for item in result["evidence"]:
        content = (root / item["source_path"]).read_bytes()
        lines = content.decode("utf-8").splitlines()
        assert item["text"] == "\n".join(lines[item["start_line"] - 1:item["end_line"]])
        assert item["source_sha256"] == hashlib.sha256(content).hexdigest()
        assert item["source_sha256"] in item["evidence_id"]
        assert item["applicable_batteries"] == ["B0005"]
        assert item["source_url"] == SOURCE_URL
    assert json.loads(json.dumps(result)) == result
    assert result == retrieve_protocol_evidence("B0005", catalog_path, root=root)


@pytest.mark.parametrize("battery, expected, forbidden", [
    ("B0005", "1.4Ahr", "software crash"),
    ("B0033", "1.6Ahr", "1.4Ahr"),
    ("B0049", "software crash", "1.4Ahr"),
])
def test_wrong_experiment_endpoints_cannot_contaminate_evidence(corpus, battery, expected, forbidden):
    root, catalog_path, _ = corpus
    result = retrieve_protocol_evidence(battery, catalog_path, root=root)
    assert result["status"] == "available"
    text = "\n".join(item["text"] for item in result["evidence"])
    assert expected in text and forbidden not in text
    assert all(battery in item["applicable_batteries"] for item in result["evidence"])


def test_inapplicable_sources_are_not_read(corpus):
    root, catalog_path, _ = corpus
    catalog, _ = read_catalog(catalog_path)
    for document in catalog["documents"][1:]:
        (root / document["path"]).unlink()
    assert retrieve_protocol_evidence("B0005", catalog_path, root=root)["status"] == "available"


def test_unknown_battery_returns_no_substitute(corpus):
    root, catalog_path, _ = corpus
    result = retrieve_protocol_evidence("B9999", catalog_path, root=root)
    assert result["status"] == "unavailable"
    assert result["evidence"] == []
    assert result["issues"] == [{"reason": "no_applicable_document"}]


@pytest.mark.parametrize("battery", [None, True, 5, "../README.txt", "B00050", " B0005"])
def test_invalid_battery_rejected_before_catalog_io(tmp_path, battery):
    with pytest.raises(ValueError, match="form B0005"):
        retrieve_protocol_evidence(battery, tmp_path / "missing.json", root=tmp_path)


def test_missing_applicable_source_is_not_replaced_by_a_wrong_document(corpus):
    root, catalog_path, _ = corpus
    catalog, _ = read_catalog(catalog_path)
    (root / catalog["documents"][0]["path"]).unlink()
    result = retrieve_protocol_evidence("B0005", catalog_path, root=root)
    assert result["status"] == "unavailable" and result["evidence"] == []
    assert result["issues"][0]["reason"] == "source_missing"


def test_hash_mismatch_returns_no_stale_evidence(corpus):
    root, catalog_path, _ = corpus
    catalog, _ = read_catalog(catalog_path)
    path = root / catalog["documents"][0]["path"]
    path.write_bytes(path.read_bytes().replace(b"1.4Ahr", b"9.9Ahr"))
    result = retrieve_protocol_evidence("B0005", catalog_path, root=root)
    assert result["status"] == "unavailable" and not result["evidence"]
    assert "hash mismatch" in result["issues"][0]["detail"]


def test_catalog_mislabel_cannot_change_source_applicability(corpus):
    root, catalog_path, _ = corpus
    catalog, _ = read_catalog(catalog_path)
    catalog["documents"][0]["applicable_batteries"] = ["B9999"]
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    result = retrieve_protocol_evidence("B9999", catalog_path, root=root)
    assert result["status"] == "unavailable" and not result["evidence"]
    assert "Files section" in result["issues"][0]["detail"]


def test_incomplete_source_sections_cannot_produce_partial_citations(corpus):
    root, catalog_path, _ = corpus
    catalog, _ = read_catalog(catalog_path)
    document = catalog["documents"][0]
    path = root / document["path"]
    content = path.read_bytes().replace(b"for discharge the fields are:", b"unknown fields:")
    path.write_bytes(content)
    register_bytes(document, content)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    result = retrieve_protocol_evidence("B0005", catalog_path, root=root)
    assert result["status"] == "unavailable" and not result["evidence"]
    assert result["issues"][0]["reason"] == "source_invalid"


def test_one_missing_source_preserves_valid_evidence_but_marks_partial(corpus):
    root, catalog_path, _ = corpus
    catalog, _ = read_catalog(catalog_path)
    missing = {**catalog["documents"][0], "document_id": "missing-supplement", "path": "data/raw/nasa/missing.txt"}
    catalog["documents"].append(missing)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    result = retrieve_protocol_evidence("B0005", catalog_path, root=root)
    assert result["status"] == "partial" and len(result["evidence"]) == 2
    assert result["issues"][0]["document_id"] == "missing-supplement"


def test_changed_source_revision_changes_evidence_identity(corpus):
    root, catalog_path, _ = corpus
    original = retrieve_protocol_evidence("B0005", catalog_path, root=root)
    catalog, _ = read_catalog(catalog_path)
    document = catalog["documents"][0]
    path = root / document["path"]
    content = path.read_bytes().replace(b"1.4Ahr", b"1.3Ahr")
    path.write_bytes(content)
    register_bytes(document, content)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    revised = retrieve_protocol_evidence("B0005", catalog_path, root=root)
    assert revised["status"] == "available"
    assert original["catalog_sha256"] != revised["catalog_sha256"]
    assert original["evidence"][0]["evidence_id"] != revised["evidence"][0]["evidence_id"]


@pytest.mark.parametrize("change, message", [
    ("duplicate", "unique"), ("path", "inside data/raw/nasa"),
])
def test_registry_errors_fail_clearly(corpus, change, message):
    root, catalog_path, _ = corpus
    catalog, _ = read_catalog(catalog_path)
    if change == "duplicate":
        catalog["documents"].append(catalog["documents"][0])
    else:
        catalog["documents"][0]["path"] = "../outside.txt"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        retrieve_protocol_evidence("B0005", catalog_path, root=root)


def test_preparation_copies_exact_cache_bytes_and_verifies_repeat(corpus, tmp_path, monkeypatch):
    _, catalog_path, cache = corpus
    destination_root = tmp_path / "prepared"
    monkeypatch.setattr(fetch_protocols, "RemoteZip", lambda: pytest.fail("No network expected"))
    paths = fetch_protocols.prepare_protocol_sources(cache, catalog_path=catalog_path, root=destination_root)
    catalog, _ = read_catalog(catalog_path)
    assert len(paths) == 3
    for document, path in zip(catalog["documents"], paths):
        assert path.read_bytes() == (cache / document["cache_member"]).read_bytes()
    assert paths == fetch_protocols.prepare_protocol_sources(catalog_path=catalog_path, root=destination_root)


def test_missing_explicit_cache_never_falls_back_to_network(corpus, tmp_path, monkeypatch):
    _, catalog_path, _ = corpus
    monkeypatch.setattr(fetch_protocols, "RemoteZip", lambda: pytest.fail("No network fallback allowed"))
    with pytest.raises(FileNotFoundError):
        fetch_protocols.prepare_protocol_sources(tmp_path / "missing", catalog_path=catalog_path, root=tmp_path / "new")


def test_preparation_rejects_bad_cache_without_creating_destination(corpus, tmp_path):
    _, catalog_path, cache = corpus
    catalog, _ = read_catalog(catalog_path)
    document = catalog["documents"][0]
    (cache / document["cache_member"]).write_bytes(b"changed")
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch_protocols.prepare_protocol_sources(cache, catalog_path=catalog_path, root=root)
    assert not (root / document["path"]).exists()


def test_preparation_does_not_overwrite_an_existing_changed_source(corpus):
    root, catalog_path, cache = corpus
    catalog, _ = read_catalog(catalog_path)
    path = root / catalog["documents"][0]["path"]
    path.write_bytes(b"user-modified source")
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch_protocols.prepare_protocol_sources(cache, catalog_path=catalog_path, root=root)
    assert path.read_bytes() == b"user-modified source"


def test_archive_path_reads_exact_registered_member_without_extractall(corpus, tmp_path, monkeypatch):
    _, catalog_path, cache = corpus
    catalog, _ = read_catalog(catalog_path)
    outer_bytes = io.BytesIO()
    with ZipFile(outer_bytes, "w") as outer:
        for document in catalog["documents"]:
            package_bytes = io.BytesIO()
            with ZipFile(package_bytes, "w") as package:
                package.writestr(document["archive_member"], (cache / document["cache_member"]).read_bytes())
                package.writestr("../unregistered.txt", b"must never extract")
            outer.writestr(document["nested_package"], package_bytes.getvalue())
    monkeypatch.setattr(fetch_protocols, "RemoteZip", lambda: io.BytesIO(outer_bytes.getvalue()))
    root = tmp_path / "from-archive"
    paths = fetch_protocols.prepare_protocol_sources(catalog_path=catalog_path, root=root)
    assert len(paths) == 3
    assert {p for p in root.rglob("*") if p.is_file()} == set(paths)


def test_langchain_protocol_tool_uses_only_battery_input(corpus):
    pytest.importorskip("langchain_ollama", reason="Optional model dependency layer")
    from scripts.model_tools import make_protocol_tool

    root, catalog_path, _ = corpus
    tool = make_protocol_tool(catalog_path, root=root)
    assert set(tool.args_schema.model_fields) == {"battery_id"}
    assert tool.invoke({"battery_id": "B0005"})["status"] == "available"


@pytest.mark.parametrize("arguments", [
    {"battery_id": "B0005", "catalog_path": "other.json"},
    {"battery_id": "B0005", "source_path": "other.txt"},
    {"battery_id": "B0005", "applicable_batteries": ["B0033"]},
    {"battery_id": 5}, {"battery_id": "../README.txt"},
])
def test_langchain_protocol_schema_rejects_source_control(arguments):
    pytest.importorskip("langchain_ollama", reason="Optional model dependency layer")
    from pydantic import ValidationError
    from scripts.model_tools import make_protocol_tool

    with pytest.raises(ValidationError):
        make_protocol_tool().invoke(arguments)
