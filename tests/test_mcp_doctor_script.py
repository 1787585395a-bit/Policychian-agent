from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.mcp_doctor import main


class MCPDoctorScriptTests(unittest.TestCase):
    def test_doctor_returns_zero_for_valid_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pythonpath = root / "src"
            pythonpath.mkdir()
            config = root / "mcp.json"
            config.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "local": {
                                "type": "stdio",
                                "command": "python",
                                "args": ["-m", "example"],
                                "env": {"PYTHONPATH": str(pythonpath)},
                                "cwd": str(root),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(main(["--mcp-config", str(config), "--json"]), 0)

    def test_doctor_returns_one_for_missing_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "mcp.json"
            config.write_text(
                json.dumps({"mcpServers": {"bad": {"type": "stdio", "command": str(root / "missing.exe")}}}),
                encoding="utf-8",
            )

            self.assertEqual(main(["--mcp-config", str(config)]), 1)


if __name__ == "__main__":
    unittest.main()
