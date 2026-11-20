# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import sys
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestManifestAndInit(unittest.TestCase):
    def test_manifest_fields_and_license(self):
        manifest_path = ROOT / "kuchisync" / "blender_manifest.toml"
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertEqual(data["id"], "kuchisync")
        self.assertEqual(data["name"], "KuchiSync")
        self.assertEqual(data["type"], "add-on")
        self.assertEqual(data["blender_version_min"], "4.2.0")
        self.assertEqual(data["license"], ["SPDX:GPL-3.0-or-later"])
        self.assertEqual(data["tags"], ["Animation", "Rigging"])
        self.assertFalse(data["tagline"].endswith("."))
        self.assertIn("files", data["permissions"])
        self.assertIn("network", data["permissions"])
        self.assertFalse(data["permissions"]["files"].endswith("."))
        self.assertFalse(data["permissions"]["network"].endswith("."))

    def test_init_imports_cleanly_outside_blender_and_has_spdx(self):
        init_path = ROOT / "kuchisync" / "__init__.py"
        text = init_path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# SPDX-License-Identifier: GPL-3.0-or-later"))
        import kuchisync

        self.assertEqual(kuchisync.bl_info["name"], "KuchiSync")
        self.assertEqual(kuchisync.bl_info["blender"], (4, 4, 0))
        kuchisync.register()
        kuchisync.unregister()

    def test_no_absolute_addon_imports(self):
        addon_dir = ROOT / "kuchisync"
        for py_path in sorted(addon_dir.glob("*.py")):
            src = py_path.read_text(encoding="utf-8")
            self.assertNotRegex(
                src,
                r"(?m)^\s*(?:from|import)\s+kuchisync\b",
                f"Extension modules must use relative imports: {py_path.name}",
            )


if __name__ == "__main__":
    unittest.main()
