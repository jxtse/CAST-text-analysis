# CAST: Achieving Stable LLM-based Text Analysis for Data Analytics

Official code, prompts, and data for our **ACL 2026** paper:

> **CAST: Achieving Stable LLM-based Text Analysis for Data Analytics**  
> Jinxiang Xie\*, Zihao Li\*, Wei He\*, Rui Ding†, Shi Han, Dongmei Zhang  
> *ACL 2026*  
> \*Equal contribution. †Corresponding author.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Text analysis of tabular data relies on two core operations:

- **Summarization** — corpus-level theme extraction
- **Tagging** — row-level labeling

A critical limitation of using LLMs for these tasks is their inability to meet
the high standards of *output stability* demanded by data analytics. We
introduce **CAST** (**C**onsistency via **A**lgorithmic Prompting and
**S**table **T**hinking), a framework that enhances output stability by
constraining the model's latent reasoning path. CAST combines:

1. **Algorithmic Prompting (AP)** — a procedural scaffold over valid reasoning
   transitions.
2. **Thinking-before-Speaking (TbS)** — explicit intermediate commitments
   before final generation.

To measure progress, we also introduce **CAST-S** and **CAST-T**, stability
metrics for bulleted summarization and tagging, validated against human
judgments. Across multiple LLM backbones, CAST consistently achieves the best
stability among all baselines, improving the Stability Score by up to
**16.2%**, while maintaining or improving output quality.

---

## Repository Layout

```
.
├── summarization/                 # Summarization (CAST-S) experiments
│   ├── summary_pipeline.py        # End-to-end summary generation + scoring
│   ├── llm_stability_pipeline.py  # Compare baseline / AP / TbS / CAST stability
│   ├── path_stability_pipeline.py # Reasoning-path ablations
│   ├── distribution_analysis_pipeline.py # Output-distribution sharpness analysis
│   ├── AblationPrompt/            # baseline / ap / tbs / cast prompts
│   ├── reasoning_path_prompt/     # Reasoning-path ablation prompts
│   ├── EvaluationPrompt/          # Judge prompts for summary + stability
│   ├── Input/                     # Input datasets (xlsx)
│   │   ├── Summary-Input/         # Datasets for summary_pipeline.py
│   │   └── Stability-Input/       # Datasets for *_stability_pipeline.py
│   └── Output/Stability-Output/   # Reference stability scores + correlation analysis
│
├── tagging/                       # Tagging (CAST-T) experiments
│   ├── program.py                 # Tagging pipeline
│   ├── evaluation.ipynb           # Evaluation / analysis notebook
│   ├── AP.md / TbS.md / AP+TbS.md / none.md   # Prompt variants
│   └── CombinedDataset.xlsx       # Tagging dataset
│
├── requirements.txt
├── .env.example                   # Copy to .env and fill in API keys
├── LICENSE                        # MIT
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/jxtse/CAST-text-analysis.git
cd CAST-text-analysis

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Python ≥ 3.9 recommended. `sentence-transformers` and `umap-learn` are only
> needed if you run `distribution_analysis_pipeline.py`.

### 2. Configure API keys

Copy the template and fill in whichever providers you intend to use:

```bash
cp .env.example .env
```

The pipelines read keys from environment variables (loaded via `python-dotenv`):

| Variable                | Used by                                  |
| ----------------------- | ---------------------------------------- |
| `OPENAI_API_KEY`        | `summary_pipeline`, `path_stability`, `tagging` |
| `OPENROUTER_API_KEY`    | All summarization pipelines              |
| `SiliconFlow_API_KEY` / `SILICONFLOW_API_KEY` | Stability + tagging pipelines |
| `Grok_API_KEY` / `GROK_API_KEY`               | Stability + tagging pipelines |
| `Gemini_API_KEY` / `GEMINI_API_KEY`           | Stability + tagging pipelines |

You only need keys for the providers you plan to call.

---

## Quickstart

All commands assume your current directory is the corresponding subfolder
(`summarization/` or `tagging/`), since the scripts use relative paths
(e.g. `Input/...`, `Output/...`).

### Summarization

End-to-end summary generation + LLM-judge scoring:

```bash
cd summarization
python summary_pipeline.py
```

Compare stability across `baseline / ap / tbs / cast` prompts:

```bash
python llm_stability_pipeline.py                       # all four
python llm_stability_pipeline.py --prompt_types cast   # subset
python llm_stability_pipeline.py --compare_only        # re-aggregate existing results
python llm_stability_pipeline.py --score_only          # rescore an existing results file
```

Reasoning-path ablations:

```bash
python path_stability_pipeline.py                      # default 4 paths
python path_stability_pipeline.py --extended_cast      # full 8-path study
python path_stability_pipeline.py --prompt_types perspective_prompt,domain_prompt
```

Output-distribution sharpness analysis:

```bash
python distribution_analysis_pipeline.py
```

### Tagging

```bash
cd tagging
# Edit the `dataset_path`, `sheet_names`, `llm_types`, and `prompt_files` lists
# at the bottom of program.py (in `async def main`) to control the run.
python program.py
```

> Note: the default `dataset_path` in `program.py` is
> `Output/Stability/CombinedDataset.xlsx`. The dataset shipped here lives at
> `tagging/CombinedDataset.xlsx`; either move/symlink it or edit the path.

Then open `evaluation.ipynb` for the post-hoc tagging analysis.

---

## Reproducing Paper Results

The reference outputs that back the figures and tables in the paper live under
`summarization/Output/Stability-Output/`:

- `cast_stability_score_result.json` — model-judged CAST-S scores
- `human_cast_stability_score_result_anonymous.json` — anonymized human ratings
- `correlation_analysis.py` — Pearson/Spearman correlation between the two
- `stability_correlation_analysis_overall.png`,
  `stability_correlation_analysis_pair.png` — corresponding figures

To regenerate the correlation figures:

```bash
cd summarization/Output/Stability-Output
python correlation_analysis.py
```

---

## Citation

If CAST or CAST-S/CAST-T are useful in your work, please cite:

```bibtex
@inproceedings{xie2026cast,
  title     = {CAST: Achieving Stable LLM-based Text Analysis for Data Analytics},
  author    = {Xie, Jinxiang and Li, Zihao and He, Wei and Ding, Rui and Han, Shi and Zhang, Dongmei},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

This work was conducted in part during Jinxiang Xie's internship at Microsoft.
Correspondence: `juding@microsoft.com`.

Issues and pull requests are welcome.
