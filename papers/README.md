# papers/

welleng's authored papers and their artifacts.

## Layout — drafts are local-only until published

- **`drafts/`** — work-in-progress papers. **Gitignored / local-only.** Drafts must
  not reach GitHub before publication (scooping / premature disclosure). Write new
  papers here.
- **`published/`** — papers that have a published **Zenodo DOI**. **Moving a paper
  from `drafts/` to `published/` is the publish step** — a deliberate, tracked commit,
  never an accidental one.
- `*-NOTES.md` working notes anywhere under `papers/` are gitignored too.

So a draft can only reach GitHub by an explicit `git mv drafts/<paper> published/` —
two layers of defence (the ignore, and the deliberate move).

The authoring → PDF (pandoc + tectonic) → Zenodo DOI workflow lives in
`docs/dev/PAPER_PIPELINE.md`; the reusable depositor is `papers/deposit-to-zenodo.py`.
PDFs build with `--resource-path=papers` so figures under `papers/figures/` resolve
regardless of where the `.md` sits.
