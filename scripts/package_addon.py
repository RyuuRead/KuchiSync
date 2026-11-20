# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
ADDON_DIR = ROOT / "kuchisync"
DEFAULT_OUTPUT = ROOT / "dist" / "kuchisync.zip"


def build_extension_zip(output_path: pathlib.Path = DEFAULT_OUTPUT) -> pathlib.Path:
    """Package kuchisync Python modules and blender_manifest.toml at the ZIP root."""
    target = pathlib.Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    py_files = [
        path
        for path in sorted(ADDON_DIR.glob("*.py"))
        if "__pycache__" not in path.parts
    ]
    manifest_file = ADDON_DIR / "blender_manifest.toml"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing extension manifest: {manifest_file}")

    bundle_files = sorted([*py_files, manifest_file], key=lambda p: p.name)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in bundle_files:
            zf.write(file_path, arcname=file_path.name)

    return target


def main() -> int:
    """CLI entry point building dist/kuchisync.zip."""
    out = build_extension_zip(DEFAULT_OUTPUT)
    print(f"Built {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
