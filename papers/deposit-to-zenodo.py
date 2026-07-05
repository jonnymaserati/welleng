#!/usr/bin/env python3
"""Deposit a paper to Zenodo via the REST API and mint a DOI.

The token is read from the ``ZENODO_TOKEN`` environment variable (export it in
your shell / ~/.bashrc), falling back to a ``ZENODO_TOKEN=...`` line in a local
``.env`` (which must be gitignored). It is never hardcoded or committed.

Usage
-----
    export ZENODO_TOKEN=...                      # scopes: deposit:write, deposit:actions
    python papers/deposit-to-zenodo.py META.zenodo.json FILE [FILE ...] [--publish] [--sandbox]

Without ``--publish`` it creates the draft, uploads the files, applies the
metadata, and STOPS — printing the draft URL for review. Re-run with
``--publish`` to mint the DOI (irreversible). Use ``--sandbox`` to dry-run
against sandbox.zenodo.org (needs a separate sandbox account + token).

See docs/dev/PAPER_PIPELINE.md.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys


def load_token() -> str:
    tok = os.environ.get("ZENODO_TOKEN")
    if tok:
        return tok.strip()
    env = pathlib.Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("ZENODO_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ZENODO_TOKEN not set (export it, or put ZENODO_TOKEN=... in .env)")


def main() -> None:
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) < 2:
        sys.exit("usage: deposit-to-zenodo.py META.zenodo.json FILE [FILE ...] "
                 "[--publish] [--sandbox]")
    meta_path, files = pos[0], pos[1:]
    publish = "--publish" in flags
    base = ("https://sandbox.zenodo.org/api" if "--sandbox" in flags
            else "https://zenodo.org/api")

    try:
        import requests
    except ImportError:
        sys.exit("needs 'requests' -> uv pip install --python .venv312 requests")

    token = load_token()
    params = {"access_token": token}

    meta = json.loads(pathlib.Path(meta_path).read_text())
    # guard: capture the version so Zenodo shows "1.0" (not its default "v1").
    # Every welleng paper sets version; the MTI + CLC deposits once dropped it and
    # displayed "v1" inconsistently with the others.
    if not meta.get("version"):
        meta["version"] = "1.0"
        print("  ! no 'version' in metadata — defaulting to 1.0 "
              "(set it explicitly in the .zenodo.json to override)")
    # guard: a placeholder/blank ORCID makes the API reject the deposit
    for c in meta.get("creators", []):
        orc = c.get("orcid", "")
        if not orc or "REPLACE" in orc.upper():
            c.pop("orcid", None)
            print(f"  ! dropped missing/placeholder ORCID for {c.get('name')!r} "
                  "— add it to the metadata for author credit")

    # 1. create the draft deposition
    r = requests.post(f"{base}/deposit/depositions", params=params, json={})
    r.raise_for_status()
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print(f"draft created: id={dep_id}")

    # 2. upload files to the bucket
    for f in files:
        fp = pathlib.Path(f)
        if not fp.exists():
            sys.exit(f"file not found: {fp}")
        with fp.open("rb") as fh:
            r = requests.put(f"{bucket}/{fp.name}", data=fh, params=params)
        r.raise_for_status()
        print(f"  uploaded {fp.name}")

    # 3. apply metadata
    r = requests.put(f"{base}/deposit/depositions/{dep_id}",
                     params=params, json={"metadata": meta})
    r.raise_for_status()
    print("  metadata applied")

    draft_url = dep["links"].get("html", f"{base.rsplit('/api', 1)[0]}/deposit/{dep_id}")
    if not publish:
        print(f"\nDRAFT ready (NOT published): {draft_url}")
        print("review it, then re-run with --publish to mint the DOI.")
        return

    # 4. publish (irreversible)
    r = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish",
                      params=params)
    r.raise_for_status()
    out = r.json()
    print(f"\nPUBLISHED -> DOI: {out.get('doi')}")
    print(f"  concept DOI (all versions): {out.get('conceptdoi')}")
    print(f"  record: {out.get('links', {}).get('record_html')}")


if __name__ == "__main__":
    main()
