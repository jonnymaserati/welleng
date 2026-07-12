# papers/

Published welleng papers and their reproducibility artifacts.

- **`published/`** — papers with a published Zenodo DOI: Markdown source + `.zenodo.json` metadata.
- **`figures/`, `data/`** — figures and validation data behind the published results.
- **`generate_*.py`** — scripts that regenerate those figures and data.

PDFs build with `--resource-path=papers` so `figures/` resolve regardless of where the `.md` sits.
