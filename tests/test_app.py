import os
import unittest
from unittest.mock import patch

import app


class AppServerConfigTest(unittest.TestCase):
    def test_main_uses_render_port_and_host(self):
        with patch.dict(os.environ, {"PORT": "8080"}, clear=False):
            with patch("app.app.run") as mock_run:
                app.main()
                mock_run.assert_called_once_with(
                    host="0.0.0.0",
                    port=8080,
                    debug=False,
                )

    def test_main_falls_back_to_default_port(self):
        with patch.dict(os.environ, {}, clear=False):
            with patch("app.app.run") as mock_run:
                app.main()
                mock_run.assert_called_once_with(
                    host="0.0.0.0",
                    port=5000,
                    debug=False,
                )


if __name__ == "__main__":
    unittest.main()
