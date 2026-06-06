import pytest

from scripts.render_cloud_run_deploy import (
    DeployConfig,
    build_deploy_plan,
    render_deploy_command,
    render_invoker_command,
    render_runtime_iam_commands,
    write_deploy_plan,
)


def _command(config: DeployConfig) -> list[str]:
    return render_deploy_command(config)


def test_deploy_command_requires_authenticated_gcs_backed_service():
    cmd = _command(
        DeployConfig(
            project="p",
            image="us-docker.pkg.dev/p/apps/delf:abc123",
            service_account="delf-runtime@p.iam.gserviceaccount.com",
            gcs_bucket="delf-corpus-prod",
        )
    )
    joined = " ".join(cmd)

    assert "--no-allow-unauthenticated" in cmd
    assert "--allow-unauthenticated" not in cmd
    assert "--service-account delf-runtime@p.iam.gserviceaccount.com" in joined
    assert "PERSISTENCE_BACKEND=gcs" in joined
    assert "GCS_BUCKET=delf-corpus-prod" in joined
    assert "GCS_SNAPSHOT_PREFIX=tee-corpus" in joined
    assert "REQUIRE_CORPUS_ON_STARTUP=true" in joined
    assert "SYNC_ON_STARTUP=true" in joined
    assert "MOCK_MODE=false" in joined
    assert "INFERENCE_BACKEND=vertex" in joined
    assert "CHROMA_DIR=/tmp/chroma_db" in joined
    assert "EMBEDDING_DIMENSION=3072" in joined


def test_deploy_command_rejects_embedding_dimension_drift():
    with pytest.raises(SystemExit, match="EMBEDDING_DIMENSION must remain 3072"):
        _command(
            DeployConfig(
                project="p",
                image="img:abc123",
                service_account="sa@p.iam.gserviceaccount.com",
                gcs_bucket="bucket",
                embedding_dimension=1024,
            )
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"memory": "1024Mi"}, "--memory must be at least 2Gi"),
        ({"cpu": "1"}, "--cpu must be at least 2"),
        ({"timeout": "120"}, "--timeout must be at least 300 seconds"),
        ({"min_instances": 0}, "--min-instances must be at least 1"),
    ],
)
def test_deploy_command_rejects_undersized_serving_shape(kwargs, message):
    config = DeployConfig(
        project="p",
        image="us-docker.pkg.dev/p/apps/delf:abc123",
        service_account="sa@p.iam.gserviceaccount.com",
        gcs_bucket="bucket",
        **kwargs,
    )

    with pytest.raises(SystemExit, match=message):
        _command(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image", "us-docker.pkg.dev/p/apps/delf:REAL_IMAGE_TAG", "--image must be a real non-placeholder value"),
        ("gcs_bucket", "DELF_CORPUS_BUCKET", "--gcs-bucket must be a real non-placeholder value"),
        ("gcs_bucket", "gs://delf-corpus-prod", "--gcs-bucket must be a bucket name, not a gs:// URI"),
        ("snapshot_prefix", "REAL_SNAPSHOT_PREFIX", "--snapshot-prefix must be a real non-placeholder value"),
        ("service_account", "serviceAccount:YOUR_SERVICE_ACCOUNT", "--service-account must be a real non-placeholder value"),
    ],
)
def test_deploy_command_rejects_placeholder_values(field, value, message):
    kwargs = {
        "project": "p",
        "image": "us-docker.pkg.dev/p/apps/delf:abc123",
        "service_account": "sa@p.iam.gserviceaccount.com",
        "gcs_bucket": "bucket",
    }
    kwargs[field] = value

    with pytest.raises(SystemExit, match=message):
        _command(DeployConfig(**kwargs))


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ("us-docker.pkg.dev/p/apps/delf", "--image must include an immutable digest or a non-latest tag"),
        ("us-docker.pkg.dev/p/apps/delf:latest", "--image must not use the mutable latest tag"),
    ],
)
def test_deploy_command_rejects_unpinned_image_references(image, message):
    with pytest.raises(SystemExit, match=message):
        _command(
            DeployConfig(
                project="p",
                image=image,
                service_account="sa@p.iam.gserviceaccount.com",
                gcs_bucket="bucket",
            )
        )


def test_deploy_command_accepts_digest_image_reference():
    cmd = _command(
        DeployConfig(
            project="p",
            image="us-docker.pkg.dev/p/apps/delf@sha256:" + "a" * 64,
            service_account="sa@p.iam.gserviceaccount.com",
            gcs_bucket="bucket",
        )
    )

    assert "us-docker.pkg.dev/p/apps/delf@sha256:" + "a" * 64 in cmd


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"image": "us-central1-docker.pkg.dev/other/apps/delf:abc123"},
            "--image Artifact Registry project must match --project",
        ),
        (
            {"service_account": "sa@other.iam.gserviceaccount.com"},
            "--service-account project must match --project",
        ),
    ],
)
def test_deploy_command_rejects_cross_project_runtime_references(kwargs, message):
    params = {
        "project": "p",
        "image": "us-central1-docker.pkg.dev/p/apps/delf:abc123",
        "service_account": "sa@p.iam.gserviceaccount.com",
        "gcs_bucket": "bucket",
    }
    params.update(kwargs)

    with pytest.raises(SystemExit, match=message):
        _command(DeployConfig(**params))


def test_invoker_command_is_explicit_and_optional():
    config = DeployConfig(
        project="p",
        image="img",
        service_account="sa@p.iam.gserviceaccount.com",
        gcs_bucket="bucket",
    )
    assert render_invoker_command(config) is None

    cmd = render_invoker_command(
        DeployConfig(
            project="p",
            image="img",
            service_account="sa@p.iam.gserviceaccount.com",
            gcs_bucket="bucket",
            invoker="group:examiners@org.test",
        )
    )
    assert cmd == [
        "gcloud",
        "run",
        "services",
        "add-iam-policy-binding",
        "delf-dalf-examiner-assistant",
        "--project",
        "p",
        "--region",
        "us-central1",
        "--member",
        "group:examiners@org.test",
        "--role",
        "roles/run.invoker",
    ]


@pytest.mark.parametrize("invoker", ["allUsers", "allAuthenticatedUsers", "group:YOUR_GROUP@example.org"])
def test_invoker_command_rejects_public_or_placeholder_invoker(invoker):
    with pytest.raises(SystemExit):
        render_invoker_command(
            DeployConfig(
                project="p",
                image="img",
                service_account="sa@p.iam.gserviceaccount.com",
                gcs_bucket="bucket",
                invoker=invoker,
            )
        )


def test_runtime_iam_commands_grant_vertex_and_bucket_read_access():
    cmd = render_runtime_iam_commands(
        DeployConfig(
            project="p",
            image="img",
            service_account="sa@p.iam.gserviceaccount.com",
            gcs_bucket="bucket",
        )
    )

    assert cmd == [
        [
            "gcloud",
            "projects",
            "add-iam-policy-binding",
            "p",
            "--member",
            "serviceAccount:sa@p.iam.gserviceaccount.com",
            "--role",
            "roles/aiplatform.user",
        ],
        [
            "gcloud",
            "storage",
            "buckets",
            "add-iam-policy-binding",
            "gs://bucket",
            "--member",
            "serviceAccount:sa@p.iam.gserviceaccount.com",
            "--role",
            "roles/storage.objectViewer",
        ],
    ]


def test_build_and_write_deploy_plan(tmp_path):
    plan = build_deploy_plan(
        DeployConfig(
            project="p",
            image="img:abc",
            service_account="sa@p.iam.gserviceaccount.com",
            gcs_bucket="bucket",
            invoker="group:team@org.test",
            certification_run_id="run-1",
        )
    )

    assert plan["metadata"]["certification_run_id"] == "run-1"
    assert plan["metadata"]["snapshot_prefix"] == "tee-corpus"
    assert plan["safety_invariants"]["private_cloud_run"] is True
    assert plan["safety_invariants"]["gcs_persistence"] is True
    assert plan["safety_invariants"]["explicit_invoker_binding"] is True
    assert plan["safety_invariants"]["runtime_vertex_access"] is True
    assert plan["safety_invariants"]["runtime_gcs_read_access"] is True
    assert plan["safety_invariants"]["serving_shape"] is True
    assert plan["resource_settings"] == {
        "memory": "2Gi",
        "cpu": "2",
        "timeout": "300",
        "min_instances": 1,
    }
    assert "runtime_iam_commands" in plan

    out = write_deploy_plan(plan, tmp_path)
    assert out.name.startswith("cloud_run_deploy_plan_")
    assert out.read_text(encoding="utf-8")
