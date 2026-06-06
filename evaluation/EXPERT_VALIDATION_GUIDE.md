# DELF Expert Validation Guide

Review file: `evaluation/delf_questions_expert_review.csv`

Source catalog: `evaluation/delf_source_catalog.csv`

If you received a ZIP, these files are in `delf_expert_validation_packet.zip`.
The ZIP also includes `delf_questions.json`, `corpus_files.json`, and a packet
manifest with checksums so engineering can verify the reviewed CSV matches the
same eval/corpus catalog that was sent.

The goal is to certify that the retrieval evaluation set is fair and grounded in
the official DELF/DALF corpus before using M3/M4 as production gates.

## What To Review

For each row, check:

- `question`: realistic examiner-training question.
- `ground_truth`: correct answer according to official DELF/DALF material.
- `expected_sources`: source files that should be retrieved to answer the question.

Use `delf_source_catalog.csv` when correcting sources. It lists every valid
corpus filename plus its document type, level, skill, and chunk count.

## Fill These Columns

Use one value in `validation_decision`:

- `VALIDATED`: question, answer, and sources are correct.
- `FIX_SOURCE`: sources need correction; fill `expert_corrected_sources`.
- `FIX_ANSWER`: answer needs correction; fill `expert_corrected_ground_truth`.
- `FIX_BOTH`: both answer and sources need correction; fill both correction columns.
- `DROP`: remove this item from certification scoring.

Use semicolons for multiple corrected sources:

```text
manuel-exacor.pdf; Diaporama_rôle_EC_VF.pptx
```

Use `expert_notes` for any ambiguity, caveat, or reason for dropping a question.

## After Review

First check the completed CSV:

```bash
python3 -m scripts.lint_expert_review \
  --review evaluation/delf_questions_expert_review.csv \
  --require-complete
```

Then apply it with:

```bash
python3 -m scripts.apply_expert_validation \
  --review evaluation/delf_questions_expert_review.csv \
  --output evaluation/delf_questions.validated.json \
  --strict
```

Then use `evaluation/delf_questions.validated.json` for the enriched retrieval
experiment and citation-grounding measurement.
