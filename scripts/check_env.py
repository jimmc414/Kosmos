#!/usr/bin/env python3
"""Report whether this environment can run `kosmos run` (VIAB#P3-2).

Prints one row per check: the default LLM provider and model, whether litellm imports,
whether the Docker daemon answers, whether sentence_transformers imports (optional: novelty
falls back to TF-IDF without it), the database URL with its password masked, and the pinned
sandbox image tag with whether that image is present locally. Reads the configuration the
CLI reads (.env and the environment) without creating directories, opening the database or
calling an LLM. Never prints a credential value.

Usage: python scripts/check_env.py
Exit 0 when every required row is OK; 1 otherwise (sentence_transformers is informational).
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mask_db_url(url):
    """Return the database URL with any password replaced by ***."""
    try:
        from sqlalchemy.engine import make_url

        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return "<unparseable DATABASE_URL>"


def check_import(module):
    """Return (ok, detail) for importing a module by name."""
    try:
        importlib.import_module(module)
        return True, "importable"
    except Exception as e:
        return False, f"not importable ({type(e).__name__})"


def check_docker(image):
    """Return (daemon_ok, daemon_detail, image_ok, image_detail)."""
    try:
        import docker
    except ImportError:
        return False, 'docker package missing (pip install "kosmos-ai-scientist[execution]")', False, "unknown"
    try:
        client = docker.from_env(timeout=5)
        client.ping()
    except Exception as e:
        return False, f"daemon unreachable ({type(e).__name__})", False, "unknown"
    try:
        client.images.get(image)
        return True, "reachable", True, "present"
    except Exception as e:
        return True, "reachable", False, f"missing ({type(e).__name__}); build docker/sandbox"


def collect_checks():
    """Return a list of (name, detail, ok, required) rows."""
    rows = []
    try:
        from kosmos.config import KosmosConfig

        config = KosmosConfig()
    except Exception as e:
        # First line only: a pydantic error's later lines echo input values, which may be keys
        first_line = str(e).splitlines()[0] if str(e) else ""
        rows.append(("Configuration (.env)", f"error ({type(e).__name__}: {first_line})", False, True))
        return rows

    rows.append(("LLM provider", config.llm_provider, True, True))
    try:
        model = config.get_active_model()
        rows.append(("LLM model", model, True, True))
    except Exception as e:
        rows.append(("LLM model", f"error ({type(e).__name__})", False, True))

    ok, detail = check_import("litellm")
    rows.append(("litellm", detail, ok, True))

    image = config.safety.sandbox_image
    daemon_ok, daemon_detail, image_ok, image_detail = check_docker(image)
    rows.append(("Docker daemon", daemon_detail, daemon_ok, True))
    rows.append((f"Sandbox image {image}", image_detail, image_ok, True))

    ok, detail = check_import("sentence_transformers")
    if not ok:
        detail += "; novelty uses TF-IDF"
    rows.append(("sentence_transformers (optional)", detail, ok, False))

    rows.append(("Database URL", mask_db_url(config.database.url), True, True))
    return rows


def main():
    rows = collect_checks()
    width = max(len(name) for name, _, _, _ in rows)
    for name, detail, ok, required in rows:
        mark = "OK  " if ok else ("FAIL" if required else "INFO")
        print(f"{mark}  {name.ljust(width)}  {detail}")
    required = [name for name, _, _, req in rows if req]
    failed = [name for name, _, ok, req in rows if req and not ok]
    print(f"check_env: {len(required) - len(failed)}/{len(required)} required OK"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
