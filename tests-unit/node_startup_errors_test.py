"""Tests for the custom node startup error tracking introduced for
Comfy-Org/ComfyUI-Launcher#303.

Covers:
- load_custom_node populates NODE_STARTUP_ERRORS with the correct source
  for each module_parent (custom_nodes / comfy_extras / comfy_api_nodes).
- Composite keying prevents collisions between modules with the same name
  in different sources.
- record_node_startup_error stores the expected fields.
- pyproject.toml metadata is attached when present and omitted when absent.
"""
import textwrap

import pytest

import nodes


@pytest.fixture(autouse=True)
def _clear_startup_errors():
    nodes.NODE_STARTUP_ERRORS.clear()
    yield
    nodes.NODE_STARTUP_ERRORS.clear()


def _write_broken_module(tmp_path, name: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent("""\
        # Deliberately broken module to exercise startup-error tracking.
        raise RuntimeError("boom from " + __name__)
    """))
    return str(path)


def test_record_node_startup_error_fields(tmp_path):
    err = ValueError("kaboom")
    nodes.record_node_startup_error(
        module_path=str(tmp_path / "my_pack"),
        source="custom_nodes",
        phase="import",
        error=err,
        tb="traceback-text",
    )
    assert "custom_nodes:my_pack" in nodes.NODE_STARTUP_ERRORS
    entry = nodes.NODE_STARTUP_ERRORS["custom_nodes:my_pack"]
    assert entry["source"] == "custom_nodes"
    assert entry["module_name"] == "my_pack"
    assert entry["phase"] == "import"
    assert entry["error"] == "kaboom"
    assert entry["traceback"] == "traceback-text"
    assert entry["module_path"].endswith("my_pack")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_parent",
    ["custom_nodes", "comfy_extras", "comfy_api_nodes"],
)
async def test_load_custom_node_records_source(tmp_path, module_parent):
    # `source` in the entry should be the same string as `module_parent`.
    module_path = _write_broken_module(tmp_path, "broken_pack")

    success = await nodes.load_custom_node(module_path, module_parent=module_parent)
    assert success is False

    key = f"{module_parent}:broken_pack"
    assert key in nodes.NODE_STARTUP_ERRORS, nodes.NODE_STARTUP_ERRORS
    entry = nodes.NODE_STARTUP_ERRORS[key]
    assert entry["source"] == module_parent
    assert entry["module_name"] == "broken_pack"
    assert entry["phase"] == "import"
    assert "boom from" in entry["error"]
    assert "RuntimeError" in entry["traceback"]


@pytest.mark.asyncio
async def test_load_custom_node_collision_across_sources(tmp_path):
    # Same module name registered as both a custom node and a comfy_extra;
    # composite keying should keep both entries.
    cn_dir = tmp_path / "cn"
    extras_dir = tmp_path / "extras"
    cn_dir.mkdir()
    extras_dir.mkdir()
    cn_path = _write_broken_module(cn_dir, "nodes_audio")
    extras_path = _write_broken_module(extras_dir, "nodes_audio")

    assert await nodes.load_custom_node(cn_path, module_parent="custom_nodes") is False
    assert await nodes.load_custom_node(extras_path, module_parent="comfy_extras") is False

    assert "custom_nodes:nodes_audio" in nodes.NODE_STARTUP_ERRORS
    assert "comfy_extras:nodes_audio" in nodes.NODE_STARTUP_ERRORS
    assert (
        nodes.NODE_STARTUP_ERRORS["custom_nodes:nodes_audio"]["module_path"]
        != nodes.NODE_STARTUP_ERRORS["comfy_extras:nodes_audio"]["module_path"]
    )


@pytest.mark.asyncio
async def test_load_custom_node_attaches_pyproject_metadata(tmp_path):
    pack_dir = tmp_path / "MyCoolPack"
    pack_dir.mkdir()
    (pack_dir / "__init__.py").write_text("raise RuntimeError('boom')\n")
    (pack_dir / "pyproject.toml").write_text(textwrap.dedent("""\
        [project]
        name = "comfyui-mycoolpack"
        version = "1.2.3"

        [project.urls]
        Repository = "https://github.com/example/comfyui-mycoolpack"

        [tool.comfy]
        PublisherId = "example"
        DisplayName = "My Cool Pack"
    """))

    success = await nodes.load_custom_node(str(pack_dir), module_parent="custom_nodes")
    assert success is False

    entry = nodes.NODE_STARTUP_ERRORS["custom_nodes:MyCoolPack"]
    assert "pyproject" in entry, entry
    py = entry["pyproject"]
    assert py["pack_id"] == "comfyui-mycoolpack"
    assert py["display_name"] == "My Cool Pack"
    assert py["publisher_id"] == "example"
    assert py["version"] == "1.2.3"
    assert py["repository"] == "https://github.com/example/comfyui-mycoolpack"


@pytest.mark.asyncio
async def test_load_custom_node_no_pyproject_skips_metadata(tmp_path):
    # Single-file extras-style module: no pyproject.toml exists alongside it,
    # so the entry must not contain a 'pyproject' key.
    module_path = _write_broken_module(tmp_path, "lonely")
    assert await nodes.load_custom_node(module_path, module_parent="comfy_extras") is False
    entry = nodes.NODE_STARTUP_ERRORS["comfy_extras:lonely"]
    assert "pyproject" not in entry


@pytest.mark.asyncio
async def test_load_custom_node_arbitrary_module_parent_passes_through(tmp_path):
    # `source` is a free-form string — an unknown module_parent (e.g. a future
    # node-source bucket) should be recorded as-is, not coerced or rejected.
    module_path = _write_broken_module(tmp_path, "future_pack")
    assert await nodes.load_custom_node(module_path, module_parent="future_source") is False
    entry = nodes.NODE_STARTUP_ERRORS["future_source:future_pack"]
    assert entry["source"] == "future_source"
