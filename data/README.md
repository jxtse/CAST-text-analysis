# Supplementary Data & Human Annotations

This folder hosts data assets that complement the pipelines in
`summarization/` and `tagging/`:

```
data/
├── human_annotations/
│   ├── summarization/        # 3 annotators' stability scores (JSON)
│   └── tagging/              # 3 annotators' tagging results (xlsx, organized by prompt)
└── supplementary/            # additional input datasets used in extended experiments
```

---

## `human_annotations/summarization/`

Three independent human annotators (`h1`, `j2`, `z3`) scored the stability of
generated summaries using the same rubric encoded in
`summarization/EvaluationPrompt/stability_evaluation_prompt.md`.

| File | Annotator |
| --- | --- |
| `processed_human_cast_stability_score_result_h1.json` | h1 |
| `processed_human_cast_stability_score_result_j2.json` | j2 |
| `processed_human_cast_stability_score_result_z3.json` | z3 |

Each file is a JSON list. Every record is one (dataset, query) instance with:

```json
{
  "dataset": "...",
  "query": "...",
  "num_generations": ...,
  "num_evaluated_pairs": ...,
  "stability_score": ...,
  "semantic_score": ...,
  "position_score": ...,
  "match_ratio": ...,
  "pair_details": [...]
}
```

These files are the human side of the correlation analysis whose model-side
counterpart is
`summarization/Output/Stability-Output/cast_stability_score_result.json`.
A pre-anonymised, paper-ready merge is also shipped at
`summarization/Output/Stability-Output/human_cast_stability_score_result_anonymous.json`,
and you can reproduce the corresponding figures with:

```bash
cd summarization/Output/Stability-Output
python correlation_analysis.py
```

---

## `human_annotations/tagging/`

Three annotators (`h1` = hewei, `j1`, `j2` = jinxiang) independently labeled
the **`Amazon_100_label`** sheet of the tagging dataset after running each
prompt variant. The folder layout per annotator is:

```
annotator_<id>/
├── AP/        Amazon_100_0_results.xlsx,  Amazon_100_1_results.xlsx
├── AP+TbS/    Amazon_100_0_results.xlsx,  Amazon_100_1_results.xlsx
├── TbS/       Amazon_100_00_results.xlsx, Amazon_100_1_results.xlsx
└── none/      Amazon_100_0_results.xlsx,  Amazon_100_1_results.xlsx
```

`*_0_results.xlsx` and `*_1_results.xlsx` correspond to two independent runs
of the same prompt variant on the `Amazon_100` data, used to measure
run-to-run stability against the human reference labels.

> The shared input dataset (`CombinedDataset.xlsx`) and the per-annotator
> annotated copies have been removed from this public release because they
> contained sensitive material. Bring your own dataset with the same column
> layout to reproduce the pipeline.

`tagging/evaluation.ipynb` is the notebook used to aggregate these
annotations into the tagging stability metrics reported in the paper.

---

## `supplementary/`

Additional datasets used in extended experiments and case studies:

| File | Description |
| --- | --- |
| `Multilingual Text Summarization Datasets (4-sheet supplement).xlsx` | A 4-sheet version (`CustomerFeedback_english`, `Tweets_italian`, `Tweets_portuguese`, `ProductReview_chinese`) used for additional multilingual stability checks. The full 12-sheet version lives at `summarization/Input/Summary-Input/xlsx/Multilingual Text Summarization Datasets.xlsx`. |
| `359.xlsx` | 200 short multilingual user utterances (`id`, `lang`, `Verbatim`). |
| `364.xlsx` | 200 multilingual user feedback entries with timestamp / language / device / verbatim text. |

These are referenced in the supplementary experiments of the paper.
