# Retrieval Certification Runbook

Objective: certify retrieval quality before production by proving:

- expert validation is complete for answerable eval items,
- enriched retrieval reaches M3: mean recall@3 >= 90%,
- generated answers keep citation grounding: citation pass rate >= 95%,
- production deployment is private and GCS-backed.

## 1. Prepare Expert Review

```bash
python3 -m scripts.prepare_expert_validation
```

This command refuses to overwrite `evaluation/delf_questions_expert_review.csv`
if any expert decision, correction, or note fields are already filled. Use
`--force-overwrite` only after saving the reviewed CSV elsewhere.

Package the files:

```bash
python3 -m scripts.package_expert_validation
```

Verify the ZIP manifest before sending:

```bash
python3 -m scripts.verify_expert_packet
```

Send `evaluation/delf_expert_validation_packet.zip` to the DELF/DALF expert.
It contains:

- `evaluation/delf_questions.json`
- `evaluation/corpus_files.json`
- `evaluation/delf_questions_expert_review.csv`
- `evaluation/EXPERT_VALIDATION_GUIDE.md`
- `evaluation/delf_source_catalog.csv`
- `evaluation/delf_expert_validation_packet_manifest.json`

The packet manifest hashes the review CSV, source catalog, eval JSON, and
corpus catalog. Keep it with the returned review so the later validation
manifest can be compared against the same source-of-truth files.

## 2. Apply Expert Review

When the CSV comes back:

One-command guarded path:

```bash
python3 -m scripts.run_certification_pipeline \
  --review evaluation/delf_questions_expert_review.csv
```

Manual equivalent:

```bash
python3 -m scripts.lint_expert_review \
  --review evaluation/delf_questions_expert_review.csv \
  --require-complete
```

```bash
python3 -m scripts.apply_expert_validation \
  --review evaluation/delf_questions_expert_review.csv \
  --output evaluation/delf_questions.validated.json \
  --strict
```

This also writes `evaluation/delf_questions.validated.validation_manifest.json`
with input/review/output checksums and strict-mode apply counts.
Both lint and apply verify that retained or corrected `expected_sources` resolve
to `evaluation/corpus_files.json`; a reviewed row with a non-corpus source is
not valid certification evidence. The validation manifest also records the
`corpus_files.json` path and SHA-256; `scripts.certification_status` rejects a
validated eval if the catalog changed after review or if any retained source no
longer resolves to the catalog.

## 3. Run Controlled Enriched Re-Ingest

This writes to `chroma_db_enriched`, not the baseline `chroma_db`.

One-command live path after expert validation:

```bash
python3 -m scripts.run_certification_pipeline \
  --review evaluation/delf_questions_expert_review.csv \
  --run-enriched-ingest \
  --measure-citations \
  --write-deploy-plan \
  --write-snapshot-plan \
  --deploy-project woven-operative-491610-u6 \
  --deploy-region us-central1 \
  --deploy-image us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf-dalf-assistant:REAL_IMAGE_TAG \
  --deploy-service-account delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com \
  --deploy-gcs-bucket REAL_CORPUS_BUCKET \
  --deploy-invoker group:REAL_APPROVED_GROUP@YOUR_DOMAIN
```

The one-command path generates a shared `certification_run_id` automatically.
For the manual path, create one run ID and pass it to every M3, M4, deploy, and
snapshot artifact command:

```bash
CERT_RUN_ID="cert-$(date -u +%Y%m%dT%H%M%SZ)"
```

The guarded pipeline only writes deploy/snapshot artifacts when
`--run-enriched-ingest` and `--measure-citations` are both present. Use the
standalone renderers in section 5 for the manual path after M3/M4 evidence has
already been produced.

Manual equivalent:

```bash
python3 -m scripts.run_enriched_retrieval_experiment \
  --eval-path evaluation/delf_questions.validated.json \
  --require-validated \
  --certification-run-id "$CERT_RUN_ID" \
  --run-ingest --force --evaluate
```

`--force` is the destructive lever for the controlled experiment only: it
removes the existing non-baseline `chroma_db_enriched` directory after refusing
baseline-like targets, symlinks, and the repository root, then rebuilds from a
zero-state directory. It does not write to the baseline `chroma_db`; the
experiment summary records the before/after baseline fingerprint so
certification can prove the baseline stayed unchanged.

Gate: `recall_with_rerank.m3_recall_at_k >= 0.90` at `m3_k=3`.
`scripts.certification_status` also verifies that the artifact came from the
controlled enriched run: contextual enrichment on, paragraph chunking, 3072-dim
embeddings, cross-lingual BM25 off, clean corpus preflight, zero unvalidated
eval items, a ready non-baseline experiment DB, and unchanged baseline
`chroma_db` fingerprints before/after the enriched run. The condensed
`enriched_experiment_*.json` must also point to the full recall JSON/Markdown
artifacts; certification cross-checks the linked recall report's M3 k, recall,
question count, per-question row count, and eval path against the summary.

## 4. Measure Citation Grounding

```bash
python3 -m scripts.measure_citation_grounding \
  --eval-path evaluation/delf_questions.validated.json \
  --chroma-dir chroma_db_enriched \
  --require-validated \
  --certification-run-id "$CERT_RUN_ID"
```

Gate: `summary.citation_pass_rate >= 0.95`. `scripts.certification_status`
also verifies that M4 used `--require-validated`, covered the full selected
validated set (`--max-questions` unset), ran against a non-baseline Chroma dir,
recorded one per-question row per selected question, had summary counts/rates
that exactly match those per-question rows, had no generation errors, and did
not trip the grounding refusal gate.

## 5. Check Overall Certification

At any point, inspect the current gate status and machine-readable next actions:

```bash
python3 -m scripts.certification_status
```

When certification is incomplete, the JSON includes `next_actions` ordered by
gate, including the current blocker, owner, required evidence, and exact command
templates for the next run step.

If you are not using the one-command path above, render a Cloud Run
deployment-safety artifact with real, non-placeholder values:

```bash
python3 -m scripts.render_cloud_run_deploy \
  --project woven-operative-491610-u6 \
  --region us-central1 \
  --image us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf-dalf-assistant:REAL_IMAGE_TAG \
  --service-account delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com \
  --gcs-bucket REAL_CORPUS_BUCKET \
  --invoker group:REAL_APPROVED_GROUP@YOUR_DOMAIN \
  --certification-run-id "$CERT_RUN_ID" \
  --write-plan
```

Also render the corpus snapshot publication artifact for the same certified
enriched Chroma directory and GCS destination:

```bash
python3 -m scripts.render_snapshot_publish_plan \
  --chroma-dir chroma_db_enriched \
  --gcs-bucket REAL_CORPUS_BUCKET \
  --snapshot-prefix tee-corpus \
  --certification-run-id "$CERT_RUN_ID" \
  --write-plan
```

The snapshot-plan renderer refuses an incomplete `chroma_db_enriched` directory
and records readiness evidence for `parents.json` and `bm25_index.pkl`;
certification rejects snapshot-publication artifacts without that evidence. The
publication command is `python -m scripts.publish_corpus_snapshot`, which
publishes the already-certified local corpus without running ingestion again.

The certification status checker rejects missing deploy and snapshot-publication
plans, placeholder bucket/snapshot-prefix/image/group values, mutable or missing
image tags, public invoker bindings, missing runtime IAM commands for Vertex AI
and GCS bucket reads, missing GCS persistence, and missing startup corpus
enforcement. Bucket values in runtime env/metadata must be bucket names without
`gs://`; generated IAM commands add `gs://` only where `gcloud storage` expects
it. The deploy-plan renderer also fails fast on obvious placeholder inputs
(`REAL_*`, `YOUR_*`, `DELF_CORPUS_BUCKET`, `:SHA`), missing image tags, `:latest`,
and public invokers so unsafe plans are not written accidentally.
The checker cross-checks deploy command flags, deploy env vars,
invoker binding, runtime IAM commands, snapshot-publication env vars, and plan
metadata so all deployment evidence points to the same project, image, service
account, corpus bucket, corpus snapshot prefix, region, service, and invoker
member. It also requires the serving shape needed for corpus snapshot restore:
memory at least 2 GiB, CPU at least 2, timeout at least 300 seconds, and at
least one warm instance.

It also checks artifact consistency across the run: M3 and M4 must use the same
validated eval path, M3/M4/snapshot-publication must use the same non-baseline
enriched Chroma directory, deploy and snapshot-publication must use the same GCS
bucket/prefix, and M3/M4 must use the same answerable/source-backed question
count as the certification eval file. The one-command pipeline stamps M3, M4,
deploy-plan, snapshot-publication, and pipeline-summary artifacts with a shared
`certification_run_id`; certification rejects artifacts whose run IDs are
missing or do not match.
The eval-validation gate also requires the validation manifest produced by
`apply_expert_validation`; manually edited validated JSON is not enough. The
manifest counts must exactly match the eval file: every question is either
validated or dropped, dropped rows are counted, and unchanged rows must be zero.

```bash
python3 -m scripts.certification_status \
  --eval-path evaluation/delf_questions.validated.json
```

Only `certified: true` is a production-readiness go. Any false gate is a no-go
until fixed and re-measured.
