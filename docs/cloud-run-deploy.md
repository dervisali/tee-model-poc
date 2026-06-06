# Cloud Run Deploy Runbook

This app must not be deployed as a public, local-disk-only Streamlit service.
Production serving requires Cloud Run IAM authentication and a GCS corpus
snapshot restored to local ephemeral disk at startup.

## Required Gates

Do not deploy production traffic until:

- `python -m scripts.certification_status` reports `certified: true`.
- The runtime service account has:
  - `roles/aiplatform.user` on the project.
  - `roles/storage.objectViewer` on the corpus snapshot bucket.
- The deploy command includes `--no-allow-unauthenticated`.
- The deploy image is pinned with an immutable digest or a non-`latest` release
  tag; `:latest` is not production evidence.
- The deploy command keeps the production serving shape:
  - `--memory` at least `2Gi`
  - `--cpu` at least `2`
  - `--timeout` at least `300`
  - `--min-instances` at least `1`
- Runtime env includes:
  - `PERSISTENCE_BACKEND=gcs`
  - `GCS_BUCKET=<snapshot-bucket-name>` without a `gs://` prefix
  - `GCS_SNAPSHOT_PREFIX=tee-corpus`
  - `SYNC_ON_STARTUP=true`
  - `REQUIRE_CORPUS_ON_STARTUP=true`
  - `MOCK_MODE=false`
  - `INFERENCE_BACKEND=vertex`
  - `CHROMA_DIR=/tmp/chroma_db`

## Render Deploy Commands

Use the same `CERT_RUN_ID` that was stamped on the passing M3 and M4 artifacts.

```bash
python -m scripts.render_cloud_run_deploy \
  --project woven-operative-491610-u6 \
  --region us-central1 \
  --image us-central1-docker.pkg.dev/woven-operative-491610-u6/apps/delf-dalf-assistant:REAL_IMAGE_TAG \
  --service-account delf-runtime@woven-operative-491610-u6.iam.gserviceaccount.com \
  --gcs-bucket REAL_CORPUS_BUCKET \
  --invoker group:REAL_APPROVED_GROUP@YOUR_DOMAIN \
  --certification-run-id "$CERT_RUN_ID" \
  --write-plan
```

The renderer also emits IAM commands granting the runtime service account
`roles/aiplatform.user` on the project and `roles/storage.objectViewer` on the
snapshot bucket. The deploy command keeps the service private; the invoker IAM
command grants `roles/run.invoker` only to the approved user or group.
`--write-plan` also emits
`evaluation/results/cloud_run_deploy_plan_*.json`; production certification
rejects placeholder values, public invokers, missing snapshot-prefix evidence,
and missing runtime IAM coverage.
The renderer fails fast on obvious placeholder inputs such as `REAL_IMAGE_TAG`,
`DELF_CORPUS_BUCKET`, `YOUR_*`, public invokers, missing image tags, or the
mutable `:latest` image tag. `--gcs-bucket` must be a bucket name; do not include
the `gs://` prefix in runtime configuration.
The certification checker also cross-checks the deploy command, invoker binding,
runtime IAM commands, environment variables, and plan metadata so a hand-edited
artifact cannot point different parts of the deployment at different projects,
buckets, snapshot prefixes, images, service accounts, or invoker groups. It also
rejects deploy plans that use mutable image tags or shrink memory, CPU, timeout,
or min instances below the corpus snapshot restore baseline.

## Snapshot Publication

After retrieval certification, render the snapshot publication plan, then
publish the certified local corpus snapshot from the same environment that
produced the passing M3/M4 artifacts:

```bash
python -m scripts.render_snapshot_publish_plan \
  --chroma-dir chroma_db_enriched \
  --gcs-bucket REAL_CORPUS_BUCKET \
  --snapshot-prefix tee-corpus \
  --certification-run-id "$CERT_RUN_ID" \
  --write-plan
```

The renderer refuses an incomplete `--chroma-dir`; it records readiness evidence
that `parents.json`, `bm25_index.pkl`, `chroma.sqlite3`, and Chroma HNSW index
files exist before a snapshot publication artifact can pass certification.

```bash
PERSISTENCE_BACKEND=gcs \
GCS_BUCKET=DELF_CORPUS_BUCKET \
GCS_SNAPSHOT_PREFIX=tee-corpus \
CHROMA_DIR=chroma_db_enriched \
python -m scripts.publish_corpus_snapshot
```

The publish hook writes a versioned `corpus.tar.gz`, `manifest.json`, and
`latest.json` without re-ingesting or mutating the certified local corpus. At
Cloud Run startup, `entrypoint.sh` runs `python -m
src.bootstrap`; the revision fails healthily if the corpus cannot be restored or
read.

## Post-Deploy Checks

```bash
gcloud run services describe delf-dalf-examiner-assistant \
  --project woven-operative-491610-u6 \
  --region us-central1 \
  --format='value(status.url)'

gcloud run services get-iam-policy delf-dalf-examiner-assistant \
  --project woven-operative-491610-u6 \
  --region us-central1
```

Confirm there is no `allUsers` or `allAuthenticatedUsers` invoker binding.
Then run one authenticated smoke query through the UI and inspect Cloud Run logs
for successful bootstrap output.
