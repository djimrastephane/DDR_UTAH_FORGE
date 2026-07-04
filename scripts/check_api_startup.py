from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PORT = 8731
HEALTH_URL = f"http://127.0.0.1:{PORT}/api/v1/health"
STARTUP_TIMEOUT_SECONDS = 60


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(PORT)],
        cwd=REPO_ROOT,
    )
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"uvicorn exited early with code {proc.returncode}")
            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=2) as resp:
                    body = resp.read().decode()
                    if resp.status == 200 and '"ok":true' in body.replace(" ", ""):
                        print(f"API started OK: {HEALTH_URL} -> {body}")
                        return 0
            except (urllib.error.URLError, ConnectionError) as exc:
                last_error = exc
            time.sleep(1)
        raise TimeoutError(
            f"API did not become healthy within {STARTUP_TIMEOUT_SECONDS}s: {last_error}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
