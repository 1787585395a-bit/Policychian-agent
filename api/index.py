from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import PolicyChainRequestHandler


class handler(BaseHTTPRequestHandler):
    """Vercel Python runtime entrypoint."""

    def do_GET(self) -> None:
        PolicyChainRequestHandler.do_GET(self)

    def do_POST(self) -> None:
        PolicyChainRequestHandler.do_POST(self)

    def log_message(self, format: str, *args) -> None:
        return
