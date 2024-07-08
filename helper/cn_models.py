import sys
import re
from pathlib import Path
from modules import extensions, sd_models, shared
from modules.paths import extensions_builtin_dir, extensions_dir, models_path

cn_model_module = {
    "inpaint": "inpaint_global_harmonious",
    "scribble": "t2ia_sketch_pidi",
    "lineart": "lineart_coarse",
    "openpose": "openpose_full",
    "tile": "tile_resample",
    "depth": "depth_midas",
    "temporalnet": "temporalnet",
    "ip-adapter": "ip-adapter",
}
cn_model_regex = re.compile("|".join(cn_model_module.keys()), flags=re.I)

ext_path = Path(extensions_dir)
ext_builtin_path = Path(extensions_builtin_dir)
controlnet_exists = False
controlnet_type = "standard"
controlnet_path = None
cn_base_path = ""

for extension in extensions.active():
    if not extension.enabled:
        continue
    # For cases like sd-webui-controlnet-master
    if "sd-webui-controlnet" in extension.name or "controlnet" in extension.name:
        controlnet_exists = True
        controlnet_path = Path(extension.path)
        cn_base_path = ".".join(controlnet_path.parts[-2:])
        break

if controlnet_path is not None:
    sd_webui_controlnet_path = controlnet_path.resolve().parent
    if sd_webui_controlnet_path.stem in ("extensions", "extensions-builtin"):
        target_path = str(sd_webui_controlnet_path.parent)
        if target_path not in sys.path:
            sys.path.append(target_path)

def get_cn_model_dirs() -> list[Path]:
    cn_model_dir = Path(models_path, "ControlNet")
    if controlnet_path is not None:
        cn_model_dir_old = controlnet_path.joinpath("models")
    else:
        cn_model_dir_old = None
    ext_dir1 = shared.opts.data.get("control_net_models_path", "")
    ext_dir2 = getattr(shared.cmd_opts, "controlnet_dir", "")

    dirs = [cn_model_dir]
    dirs += [
        Path(ext_dir) for ext_dir in [cn_model_dir_old, ext_dir1, ext_dir2] if ext_dir
    ]

    return dirs


def _get_cn_models(cn_model_filter) -> list[str]:
    cn_model_exts = (".pt", ".pth", ".ckpt", ".safetensors")
    dirs = get_cn_model_dirs()
    name_filter = shared.opts.data.get("control_net_models_name_filter", "")
    name_filter = name_filter.strip(" ").lower()
    cn_model_regex = re.compile(cn_model_filter, flags=re.I)

    model_paths = []

    for base in dirs:
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if (
                p.is_file()
                and p.suffix in cn_model_exts
                and cn_model_regex.search(p.name)
            ):
                if name_filter and name_filter not in p.name.lower():
                    continue
                model_paths.append(p)
    model_paths.sort(key=lambda p: p.name)

    models = []
    for p in model_paths:
        model_hash = sd_models.model_hash(p)
        name = f"{p.stem} [{model_hash}]"
        models.append(name)
    return models


def get_cn_models(cn_model_filter="temporalnet") -> list[str]:
    if controlnet_exists:
        return _get_cn_models(cn_model_filter)
    return []
