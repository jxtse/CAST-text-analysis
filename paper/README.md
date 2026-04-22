# Paper

The arXiv-ready LaTeX source of our paper lives outside this repo (it is
heavy due to a 19 MB vector pipeline figure). To rebuild or re-submit:

- Source bundle (uploaded to arXiv): `CAST_arxiv_submission.zip` (~20 MB)
- Compiled PDF: `CAST_main.pdf` (~20 MB)

Both can be regenerated from the camera-ready LaTeX by adding a single
GitHub-link footnote at the end of the abstract, then running:

```bash
tectonic --keep-intermediates main.tex
```

The `--keep-intermediates` flag preserves `main.bbl`, which arXiv requires
because it does not run BibTeX itself.

## Read it online

- 📄 **arXiv**: https://arxiv.org/abs/2602.15861
- 📚 **ACL Anthology**: *(link will be added once the proceedings are out)*

## Cite

```bibtex
@misc{xie2026castachievingstablellmbased,
  title         = {CAST: Achieving Stable LLM-based Text Analysis for Data Analytics},
  author        = {Jinxiang Xie and Zihao Li and Wei He and Rui Ding and Shi Han and Dongmei Zhang},
  year          = {2026},
  eprint        = {2602.15861},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2602.15861}
}
```
