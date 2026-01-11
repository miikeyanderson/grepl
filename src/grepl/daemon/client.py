"""Client for communicating with grepl daemon."""

import hashlib
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx

from .server import get_daemon_socket_path


class DaemonClient:
    """Client for communicating with daemon over Unix socket."""

    def __init__(self, project_path: Path):
        self.project_path = project_path.resolve()
        self.socket_path = get_daemon_socket_path(self.project_path)
        self.transport = httpx.HTTPTransport(uds=str(self.socket_path))
        self.client = httpx.Client(transport=self.transport, base_url="http://daemon")

    def is_running(self) -> bool:
        """Check if daemon is running for this project."""
        if not self.socket_path.exists():
            return False

        try:
            resp = self.client.get("/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def ensure_running(self):
        """Ensure daemon is running, start if needed."""
        if self.is_running():
            return

        self._spawn_daemon()
        self._wait_for_ready()

    def _spawn_daemon(self):
        """Spawn daemon process in background."""
        # Get path to grepl module
        import grepl
        grepl_path = Path(grepl.__file__).parent.parent

        # Start daemon process
        cmd = [
            sys.executable,
            "-m", "grepl.daemon.server",
            str(self.project_path)
        ]

        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

    def _wait_for_ready(self, timeout: float = 10.0):
        """Wait for daemon to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_running():
                return
            time.sleep(0.1)

        raise TimeoutError(f"Daemon did not start within {timeout}s")

    def search(self, query: str, limit: int = 10) -> List[dict]:
        """Search using daemon."""
        self.ensure_running()

        resp = self.client.post("/search", json={"query": query, "limit": limit}, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        """Get daemon health status."""
        resp = self.client.get("/health", timeout=2.0)
        resp.raise_for_status()
        return resp.json()

    def shutdown(self):
        """Shutdown daemon."""
        try:
            resp = self.client.post("/shutdown", timeout=2.0)
            return resp.json()
        except Exception:
            pass

    def close(self):
        """Close client connection."""
        self.client.close()


def get_socket_path(project_path: Path) -> Path:
    """Get socket path for a project."""
    return get_daemon_socket_path(project_path)


def is_daemon_running(project_path: Path) -> bool:
    """Check if daemon is running for a project."""
    client = DaemonClient(project_path)
    running = client.is_running()
    client.close()
    return running
