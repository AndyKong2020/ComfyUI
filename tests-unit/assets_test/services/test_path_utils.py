"""Tests for path_utils – asset category resolution."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.services.path_utils import (
    compute_display_name,
    compute_file_path,
    get_asset_category_and_relative_path,
    get_comfy_models_folders,
    get_name_and_tags_from_asset_path,
)


@pytest.fixture
def fake_dirs():
    """Create temporary input, output, and temp directories."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        models_dir = root_path / "models"
        checkpoints_dir = models_dir / "checkpoints"
        output_checkpoints_dir = output_dir / "checkpoints"
        external_checkpoints_dir = root_path / "external" / "not_named_like_category"
        for d in (
            input_dir,
            output_dir,
            temp_dir,
            checkpoints_dir,
            output_checkpoints_dir,
            external_checkpoints_dir,
        ):
            d.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)
            mock_fp.models_dir = str(models_dir)

            with patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[
                    (
                        "checkpoints",
                        [
                            str(checkpoints_dir),
                            str(output_checkpoints_dir),
                            str(external_checkpoints_dir),
                        ],
                    )
                ],
            ):
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "temp": temp_dir,
                    "models": models_dir,
                    "checkpoints": checkpoints_dir,
                    "output_checkpoints": output_checkpoints_dir,
                    "external_checkpoints": external_checkpoints_dir,
                }


class TestGetAssetCategoryAndRelativePath:
    def test_input_file(self, fake_dirs):
        f = fake_dirs["input"] / "photo.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "input"
        assert rel == "photo.png"

    def test_output_file(self, fake_dirs):
        f = fake_dirs["output"] / "result.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "output"
        assert rel == "result.png"

    def test_temp_file(self, fake_dirs):
        """Regression: temp files must be categorised, not raise ValueError."""
        f = fake_dirs["temp"] / "GLSLShader_output_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert rel == "GLSLShader_output_00004_.png"

    def test_temp_file_in_subfolder(self, fake_dirs):
        sub = fake_dirs["temp"] / "sub"
        sub.mkdir()
        f = sub / "ComfyUI_temp_tczip_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert os.path.normpath(rel) == os.path.normpath("sub/ComfyUI_temp_tczip_00004_.png")

    def test_model_file(self, fake_dirs):
        f = fake_dirs["checkpoints"] / "model.safetensors"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "models"

    def test_unknown_path_raises(self, fake_dirs):
        with pytest.raises(ValueError, match="not within"):
            get_asset_category_and_relative_path("/some/random/path.png")


class TestResponsePaths:
    def test_get_comfy_models_folders_excludes_core_infrastructure(self, tmp_path: Path):
        controlnet_dir = tmp_path / "models" / "controlnet"
        configs_dir = tmp_path / "models" / "configs"
        custom_nodes_dir = tmp_path / "custom_nodes"
        for directory in (controlnet_dir, configs_dir, custom_nodes_dir):
            directory.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.folder_names_and_paths = {
                "controlnet": ([str(controlnet_dir)], {".safetensors"}),
                "configs": ([str(configs_dir)], {".yaml"}),
                "custom_nodes": ([str(custom_nodes_dir)], set()),
            }

            folders = get_comfy_models_folders()

        assert folders == [("controlnet", [str(controlnet_dir)])]

    def test_input_file_path_and_display_name_include_subfolder(self, fake_dirs):
        sub = fake_dirs["input"] / "some" / "folder"
        sub.mkdir(parents=True)
        f = sub / "image.png"
        f.touch()

        assert compute_file_path(str(f)) == "input/some/folder/image.png"
        assert compute_display_name(str(f)) == "some/folder/image.png"

    def test_model_file_path_is_relative_to_physical_models_root(self, fake_dirs):
        sub = fake_dirs["checkpoints"] / "flux"
        sub.mkdir()
        f = sub / "model.safetensors"
        f.touch()

        assert compute_file_path(str(f)) == "models/checkpoints/flux/model.safetensors"
        assert compute_display_name(str(f)) == "flux/model.safetensors"

    @pytest.mark.parametrize(
        "folder_name",
        ["checkpoints", "clip", "vae", "diffusion_models", "loras"],
    )
    def test_output_model_folder_uses_model_namespace_file_path(self, fake_dirs, folder_name):
        output_model_dir = fake_dirs["output"] / folder_name
        output_model_dir.mkdir(exist_ok=True)
        default_model_dir = fake_dirs["models"] / folder_name
        default_model_dir.mkdir(exist_ok=True)
        f = output_model_dir / "saved.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[(folder_name, [str(default_model_dir), str(output_model_dir)])],
        ):
            cat, rel = get_asset_category_and_relative_path(str(f))
            assert cat == "models"
            assert os.path.normpath(rel) == os.path.normpath(f"{folder_name}/saved.safetensors")

            assert compute_file_path(str(f)) == f"models/{folder_name}/saved.safetensors"
            assert compute_display_name(str(f)) == "saved.safetensors"

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "saved.safetensors"
            assert tags[:2] == ["models", folder_name]

    def test_output_model_subfolder_uses_model_namespace_file_path(self, fake_dirs):
        folder_name = "loras"
        output_model_dir = fake_dirs["output"] / folder_name
        subdir = output_model_dir / "experiments"
        subdir.mkdir(parents=True)
        f = subdir / "my_lora.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[(folder_name, [str(output_model_dir)])],
        ):
            cat, rel = get_asset_category_and_relative_path(str(f))
            assert cat == "models"
            assert os.path.normpath(rel) == os.path.normpath(
                "loras/experiments/my_lora.safetensors"
            )

            assert (
                compute_file_path(str(f))
                == "models/loras/experiments/my_lora.safetensors"
            )
            assert compute_display_name(str(f)) == "experiments/my_lora.safetensors"

            name, tags = get_name_and_tags_from_asset_path(str(f))
            assert name == "my_lora.safetensors"
            assert tags[:3] == ["models", "loras", "experiments"]

    def test_external_model_folder_uses_registered_folder_name_namespace(self, fake_dirs):
        f = fake_dirs["external_checkpoints"] / "external.safetensors"
        f.touch()

        assert compute_file_path(str(f)) == "models/checkpoints/external.safetensors"
        assert compute_display_name(str(f)) == "external.safetensors"

    def test_nested_registered_bases_for_same_model_folder_use_deepest_match(self, tmp_path: Path):
        llm_dir = tmp_path / "models" / "LLM"
        llm_checkpoints_dir = llm_dir / "checkpoints"
        llm_checkpoints_dir.mkdir(parents=True)
        checkpoint = llm_checkpoints_dir / "model.safetensors"
        checkpoint.touch()
        tokenizer = llm_dir / "tokenizer.json"
        tokenizer.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("LLM", [str(llm_checkpoints_dir), str(llm_dir)])],
        ):
            checkpoint_cat, checkpoint_rel = get_asset_category_and_relative_path(
                str(checkpoint)
            )
            tokenizer_cat, tokenizer_rel = get_asset_category_and_relative_path(
                str(tokenizer)
            )

            assert checkpoint_cat == tokenizer_cat == "models"
            assert checkpoint_rel == "LLM/model.safetensors"
            assert tokenizer_rel == "LLM/tokenizer.json"
            assert compute_file_path(str(checkpoint)) == "models/LLM/model.safetensors"
            assert compute_display_name(str(checkpoint)) == "model.safetensors"
            assert compute_file_path(str(tokenizer)) == "models/LLM/tokenizer.json"
            assert compute_display_name(str(tokenizer)) == "tokenizer.json"

    def test_same_relative_model_file_under_multiple_roots_shares_logical_file_path(
        self, tmp_path: Path
    ):
        foo_dir = tmp_path / "foo"
        bar_dir = tmp_path / "bar"
        foo_dir.mkdir()
        bar_dir.mkdir()
        foo_file = foo_dir / "baz.safetensors"
        bar_file = bar_dir / "baz.safetensors"
        foo_file.touch()
        bar_file.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(foo_dir), str(bar_dir)])],
        ):
            assert compute_file_path(str(foo_file)) == "models/checkpoints/baz.safetensors"
            assert compute_file_path(str(bar_file)) == "models/checkpoints/baz.safetensors"
            assert compute_display_name(str(foo_file)) == "baz.safetensors"
            assert compute_display_name(str(bar_file)) == "baz.safetensors"

    def test_output_clip_folder_uses_canonical_text_encoders_folder_name(self, fake_dirs):
        output_clip_dir = fake_dirs["output"] / "clip"
        output_clip_dir.mkdir()
        f = output_clip_dir / "clip_l.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("text_encoders", [str(output_clip_dir)])],
        ):
            assert compute_file_path(str(f)) == "models/text_encoders/clip_l.safetensors"
            assert compute_display_name(str(f)) == "clip_l.safetensors"

    def test_physical_unet_folder_uses_diffusion_models_namespace(self, fake_dirs):
        unet_dir = fake_dirs["models"] / "unet"
        diffusion_models_dir = fake_dirs["models"] / "diffusion_models"
        unet_dir.mkdir()
        diffusion_models_dir.mkdir()
        f = unet_dir / "wan.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("diffusion_models", [str(unet_dir), str(diffusion_models_dir)])],
        ):
            cat, rel = get_asset_category_and_relative_path(str(f))
            assert cat == "models"
            assert rel == "diffusion_models/wan.safetensors"
            assert compute_file_path(str(f)) == "models/diffusion_models/wan.safetensors"
            assert compute_display_name(str(f)) == "wan.safetensors"

    def test_unregistered_file_under_physical_models_root_has_no_file_path(self, fake_dirs):
        f = fake_dirs["models"] / "not_registered" / "orphan.bin"
        f.parent.mkdir()
        f.touch()

        assert compute_file_path(str(f)) is None
        assert compute_display_name(str(f)) is None

    def test_unknown_path_returns_none(self, fake_dirs):
        assert compute_file_path("/some/random/path.png") is None
        assert compute_display_name("/some/random/path.png") is None
