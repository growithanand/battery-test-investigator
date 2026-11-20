"""Check local Gemini key configuration."""

import os
from pathlib import Path
import sys

from dotenv import dotenv_values


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def read_gemini_key(env_file: Path = ENV_FILE) -> str:
    """Use the process environment first, then the project's local .env file."""
    value = os.environ.get("GEMINI_API_KEY")
    if value is None:
        value = dotenv_values(env_file, encoding="utf-8-sig", interpolate=False).get("GEMINI_API_KEY")
    key = (value or "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to .env or your environment.")
    if not key.isascii() or any(character.isspace() for character in key):
        raise ValueError("GEMINI_API_KEY must be a single ASCII value without spaces.")
    return key


def main() -> int:
    try:
        read_gemini_key()
    except (OSError, UnicodeError):
        print("Could not read .env. Check the file's permissions and UTF-8 encoding.", file=sys.stderr)
        return 2
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    print("Gemini API key is available locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
