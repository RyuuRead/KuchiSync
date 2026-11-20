# SPDX-License-Identifier: GPL-3.0-or-later

import pathlib
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


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


if __name__ == "__main__":
    unittest.main()
