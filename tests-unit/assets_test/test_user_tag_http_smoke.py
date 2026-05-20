"""HTTP-layer smoke test: user-added tags via POST /api/assets/{id}/tags
land after path tags when read back via GET /api/assets.

Exercises the full route handler -> service -> query path that the unit
tests at tests-unit/assets_test/queries/test_asset_info.py only cover at
the service layer.
"""
import json

import pytest
import requests


@pytest.fixture
def smoke_asset(http: requests.Session, api_base: str):
    """Upload a single asset into models/checkpoints/unit-tests/smoke
    and delete it on teardown."""
    name = "smoke_user_tag.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "smoke"]
    files = {"file": (name, b"S" * 4096, "application/octet-stream")}
    form_data = {
        "tags": json.dumps(tags),
        "name": name,
        "user_metadata": json.dumps({}),
    }
    r = http.post(api_base + "/api/assets", files=files, data=form_data, timeout=120)
    assert r.status_code == 201, r.text
    body = r.json()
    yield body
    http.delete(
        f"{api_base}/api/assets/{body['id']}?delete_content=true", timeout=30
    )


def _fetch_asset_tags(http, api_base, ref_id):
    r = http.get(f"{api_base}/api/assets/{ref_id}", timeout=30)
    assert r.status_code == 200, r.text
    return r.json()["tags"]


def test_user_tag_lands_after_path_tags_via_http(
    http: requests.Session, api_base: str, smoke_asset: dict
):
    ref_id = smoke_asset["id"]

    initial_tags = _fetch_asset_tags(http, api_base, ref_id)
    # Path tags should already be at the front in upload order.
    assert initial_tags[:2] == ["models", "checkpoints"]

    # Add a user tag that would jump to position 0 under alphabetical sort.
    r = http.post(
        f"{api_base}/api/assets/{ref_id}/tags",
        json={"tags": ["aaa-user-tag"]},
        timeout=30,
    )
    assert r.status_code in (200, 201), r.text

    tags_after = _fetch_asset_tags(http, api_base, ref_id)
    # Path tags must still be at the front; user tag goes to the end.
    assert tags_after[0] == "models"
    assert tags_after[1] == "checkpoints"
    assert "aaa-user-tag" in tags_after
    assert tags_after[-1] == "aaa-user-tag"


def test_user_tag_batch_lands_after_path_tags_via_http(
    http: requests.Session, api_base: str, smoke_asset: dict
):
    ref_id = smoke_asset["id"]

    # Add three user tags in a single request, in non-alphabetical input
    # order. They should all land after the path tags (microsecond stagger
    # in set_reference_tags / add_tags_to_reference is what makes this
    # work — without it, "aaa" would jump to position 0).
    r = http.post(
        f"{api_base}/api/assets/{ref_id}/tags",
        json={"tags": ["zzz-z", "favorite", "aaa-experiment"]},
        timeout=30,
    )
    assert r.status_code in (200, 201), r.text

    tags_after = _fetch_asset_tags(http, api_base, ref_id)
    assert tags_after[0] == "models"
    assert tags_after[1] == "checkpoints"
    user_tail = tags_after[len({"models", "checkpoints", "unit-tests", "smoke"}):]
    assert set(user_tail) >= {"zzz-z", "favorite", "aaa-experiment"}
    # Critically: alphabetical sort would put 'aaa-experiment' at position 0.
    assert tags_after.index("aaa-experiment") > tags_after.index("models")
    assert tags_after.index("aaa-experiment") > tags_after.index("checkpoints")
