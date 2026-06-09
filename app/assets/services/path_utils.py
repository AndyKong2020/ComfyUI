import os
from pathlib import Path
from typing import Literal

import folder_paths
from app.assets.helpers import normalize_tags


# These names are bootstrapped into folder_names_and_paths by core but are not
# model folders (matching /api/experiment/models' exclusion). Intentionally
# duplicated here so the assets layer stays decoupled from the legacy
# model-manager code it will eventually replace.
_NON_MODEL_FOLDER_NAMES = frozenset({"configs", "custom_nodes"})

AssetRoot = Literal["input", "output", "temp", "models"]


def get_comfy_models_folders() -> list[tuple[str, list[str]]]:
    """Build list of (folder_name, base_paths[]) for all model locations.

    Includes every folder name registered in folder_names_and_paths,
    regardless of whether its paths are under the main models_dir,
    but excludes non-model entries like configs and custom_nodes.
    """
    targets: list[tuple[str, list[str]]] = []
    for name, values in folder_paths.folder_names_and_paths.items():
        if name in _NON_MODEL_FOLDER_NAMES:
            continue
        paths, _exts = values[0], values[1]
        if paths:
            targets.append((name, paths))
    return targets


def get_comfy_model_folder_names() -> set[str]:
    """Return valid model folder names for public asset model paths."""
    return {name for name, _paths in get_comfy_models_folders()}


def is_comfy_model_folder_name(folder_name: str) -> bool:
    """Return whether a folder name resolves to a public model folder name."""
    return folder_paths.map_legacy(folder_name) in get_comfy_model_folder_names()


def resolve_destination_from_tags(tags: list[str]) -> tuple[str, list[str]]:
    """Validates and maps tags -> (base_dir, subdirs_for_fs)"""
    if not tags:
        raise ValueError("tags must not be empty")
    root = tags[0].lower()
    if root == "models":
        if len(tags) < 2:
            raise ValueError("at least two tags required for model asset")
        folder_name = folder_paths.map_legacy(tags[1])
        if not is_comfy_model_folder_name(tags[1]):
            raise ValueError(f"unknown model category '{tags[1]}'")
        try:
            bases = folder_paths.folder_names_and_paths[folder_name][0]
        except KeyError:
            raise ValueError(f"unknown model category '{tags[1]}'")
        if not bases:
            raise ValueError(f"no base path configured for category '{tags[1]}'")
        base_dir = os.path.abspath(bases[0])
        raw_subdirs = tags[2:]
    elif root == "input":
        base_dir = os.path.abspath(folder_paths.get_input_directory())
        raw_subdirs = tags[1:]
    elif root == "output":
        base_dir = os.path.abspath(folder_paths.get_output_directory())
        raw_subdirs = tags[1:]
    else:
        raise ValueError(f"unknown root tag '{tags[0]}'; expected 'models', 'input', or 'output'")
    _sep_chars = frozenset(("/", "\\", os.sep))
    for i in raw_subdirs:
        if i in (".", "..") or _sep_chars & set(i):
            raise ValueError("invalid path component in tags")

    return base_dir, raw_subdirs if raw_subdirs else []


def validate_path_within_base(candidate: str, base: str) -> None:
    cand_abs = Path(os.path.abspath(candidate))
    base_abs = Path(os.path.abspath(base))
    if not cand_abs.is_relative_to(base_abs):
        raise ValueError("destination escapes base directory")


def compute_asset_response_paths(
    file_path: str,
) -> tuple[str, str | None] | None:
    """Compute (file_path, display_name) for an Asset response.

    `file_path` is a logical namespace key: `<root>/<rel>` for input/output/temp
    assets and `models/<folder_name>/<rel>` for files under ComfyUI's registered
    model-folder paths. `display_name` is the path below that root or registered
    folder name, suitable for UI labels. Returns None when the absolute path is
    not under a known asset root or registered model-folder path.
    """
    try:
        root, folder_name, rel = get_asset_root_folder_name_and_filepath(file_path)
    except ValueError:
        return None

    display_name = rel or None
    if folder_name is None:
        response_file_path = f"{root}/{rel}" if rel else root
    else:
        response_file_path = f"{root}/{folder_name}/{rel}" if rel else f"{root}/{folder_name}"
    return response_file_path, display_name


def compute_display_name(file_path: str) -> str | None:
    """Return the asset's `display_name`, or None for unknown paths."""
    result = compute_asset_response_paths(file_path)
    return result[1] if result else None


def compute_file_path(file_path: str) -> str | None:
    """Return the asset's logical `file_path`, or None for unknown paths."""
    result = compute_asset_response_paths(file_path)
    return result[0] if result else None


def compute_relative_filename(file_path: str) -> str | None:
    """
    Return the path relative to the asset root or model folder name, using forward slashes, eg:
      /.../models/checkpoints/flux/123/flux.safetensors -> "flux/123/flux.safetensors"
      /.../models/text_encoders/clip_g.safetensors -> "clip_g.safetensors"
      /.../input/sub/image.png -> "sub/image.png"

    For unknown paths, returns None.
    """
    return compute_display_name(file_path)


def get_asset_root_folder_name_and_filepath(
    file_path: str,
) -> tuple[AssetRoot, str | None, str]:
    """Decompose an absolute path into (root, registered folder name, path-under-root).

    `folder_name` is set only when the path is under a ComfyUI registered
    model-folder path from `folder_names_and_paths`. The returned relative path
    always uses `/` separators and is empty when the path is exactly the matched
    root.

    Raises:
        ValueError: path does not belong to any known root.
    """
    fp_abs = os.path.abspath(file_path)

    def _check_is_within(child: str, parent: str) -> bool:
        return Path(child).is_relative_to(parent)

    def _compute_relative(child: str, parent: str) -> str:
        # Normalize relative path, stripping any leading ".." components
        # by anchoring to root (os.sep) then computing relpath back from it.
        rel = os.path.relpath(
            os.path.join(os.sep, os.path.relpath(child, parent)), os.sep
        )
        return "" if rel == "." else rel.replace(os.sep, "/")

    # Registered model folders define ComfyUI's model namespace. Check these
    # first so output-backed or external model paths become
    # models/<folder_name>/<relative-path> rather than physical output/... paths.
    best_model: tuple[int, str, str] | None = None
    for folder_name, bases in get_comfy_models_folders():
        for b in bases:
            base_abs = os.path.abspath(b)
            if not _check_is_within(fp_abs, base_abs):
                continue
            cand = (len(base_abs), folder_name, _compute_relative(fp_abs, base_abs))
            if best_model is None or cand[0] > best_model[0]:
                best_model = cand

    if best_model is not None:
        _, folder_name, rel_inside = best_model
        return "models", folder_name, rel_inside

    for root_tag, getter in (
        ("input", folder_paths.get_input_directory),
        ("output", folder_paths.get_output_directory),
        ("temp", folder_paths.get_temp_directory),
    ):
        base = os.path.abspath(getter())
        if _check_is_within(fp_abs, base):
            return root_tag, None, _compute_relative(fp_abs, base)

    raise ValueError(
        f"Path is not within input, output, temp, or configured model bases: {file_path}"
    )


def get_asset_category_and_relative_path(
    file_path: str,
) -> tuple[AssetRoot, str]:
    """Determine which root category a file path belongs to.

    Categories:
      - 'input': under folder_paths.get_input_directory()
      - 'output': under folder_paths.get_output_directory()
      - 'temp': under folder_paths.get_temp_directory()
      - 'models': under any base path from get_comfy_models_folders()

    Returns:
        (root_category, relative_path_inside_that_root)

    Raises:
        ValueError: path does not belong to any known root.
    """
    root, folder_name, rel = get_asset_root_folder_name_and_filepath(file_path)
    if folder_name is None:
        return root, rel

    combined = os.path.join(folder_name, rel)
    normalized = os.path.relpath(os.path.join(os.sep, combined), os.sep)
    # Normalize to forward slashes so the logical path is identical across
    # platforms (os.path.relpath emits backslashes on Windows).
    return root, normalized.replace(os.sep, "/")


def get_name_and_tags_from_asset_path(file_path: str) -> tuple[str, list[str]]:
    """Return (name, tags) derived from a filesystem path.

    - name: base filename with extension
    - tags: [root_category] + parent folder names in order

    Raises:
        ValueError: path does not belong to any known root.
    """
    root_category, some_path = get_asset_category_and_relative_path(file_path)
    p = Path(some_path)
    parent_parts = [
        part for part in p.parent.parts if part not in (".", "..", p.anchor)
    ]
    return p.name, list(dict.fromkeys(normalize_tags([root_category, *parent_parts])))
