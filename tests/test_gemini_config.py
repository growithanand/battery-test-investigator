"""Check key precedence and make sure configuration messages keep key values private."""

import pytest

pytest.importorskip("dotenv", reason="Install the optional agent dependencies")

from scripts import gemini_config


@pytest.fixture(autouse=True)
def clear_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_environment_key_takes_priority_before_reading_file(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-environment-value")

    def fail_read(*args, **kwargs):
        pytest.fail("A configured environment key should be read without file access")

    monkeypatch.setattr(gemini_config, "dotenv_values", fail_read)
    assert gemini_config.read_gemini_key() == "synthetic-environment-value"


def test_quoted_file_value_is_literal_and_does_not_set_environment(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('GEMINI_API_KEY="synthetic-${REFERENCE}"\n', encoding="utf-8-sig")
    monkeypatch.setenv("REFERENCE", "expanded-value")
    assert gemini_config.read_gemini_key(path) == "synthetic-${REFERENCE}"
    assert "GEMINI_API_KEY" not in gemini_config.os.environ


@pytest.mark.parametrize("content", [None, "", "GEMINI_API_KEY=\n", "GEMINI_API_KEY\n"])
def test_missing_key_has_a_clear_error(tmp_path, content):
    path = tmp_path / ".env"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
        gemini_config.read_gemini_key(path)


def test_blank_environment_value_is_explicit_and_rejected(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=synthetic-file-value\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "  ")
    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
        gemini_config.read_gemini_key(path)


@pytest.mark.parametrize("key", ["synthetic value", "synthetic\nvalue", "synthetic-\u00e9"])
def test_invalid_header_values_are_rejected_without_echoing_them(monkeypatch, capsys, key):
    monkeypatch.setenv("GEMINI_API_KEY", key)
    assert gemini_config.main() == 2
    output = capsys.readouterr()
    assert "single ASCII value" in output.err
    assert key not in output.out + output.err


def test_success_message_omits_key(monkeypatch, capsys):
    key = "synthetic-environment-value"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    assert gemini_config.main() == 0
    output = capsys.readouterr()
    assert output.out == "Gemini API key is available locally.\n"
    assert not output.err and key not in output.out


def test_missing_key_message(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert gemini_config.main() == 2
    output = capsys.readouterr()
    assert not output.out
    assert "GEMINI_API_KEY is missing" in output.err


@pytest.mark.parametrize("error", [OSError("synthetic-secret"), UnicodeError("synthetic-secret")])
def test_file_error_message_omits_exception_details(monkeypatch, capsys, error):
    def fail_read(*args, **kwargs):
        raise error

    monkeypatch.setattr(gemini_config, "dotenv_values", fail_read)
    assert gemini_config.main() == 2
    output = capsys.readouterr()
    assert "Could not read .env" in output.err
    assert "synthetic-secret" not in output.out + output.err
