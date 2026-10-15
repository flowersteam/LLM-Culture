"""Canonical filesystem locations for packaged data.

A single absolute, package-relative constant avoids the previous footgun where
different entrypoints read different parameter directories depending on the
current working directory (the CLI used ``llm_culture/data/parameters`` while the
web/reproduction paths used a cwd-relative ``data/parameters``).

The parameter files ship inside the package, so their location is derived from
this file's location rather than from the process's cwd.
"""
from pathlib import Path

PARAMS_DIR = Path(__file__).resolve().parent / "data" / "parameters"
PROMPT_INIT_JSON = PARAMS_DIR / "prompt_init.json"
PROMPT_UPDATE_JSON = PARAMS_DIR / "prompt_update.json"
PERSONALITIES_JSON = PARAMS_DIR / "personalities.json"
NETWORK_STRUCTURES_JSON = PARAMS_DIR / "network_structures.json"
