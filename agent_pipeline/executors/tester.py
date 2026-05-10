from __future__ import annotations

from pathlib import Path

from ..models import TestReport
from ..utils import run_cmd


class SiteTester:
    def test(self, root: Path) -> TestReport:
        checks = []
        logs = []

        rc, out = run_cmd(["npm", "--version"], cwd=root, timeout_s=120)
        checks.append(f"npm_version_rc={rc}")
        logs.append(out)
        if rc != 0:
            return TestReport(ok=False, checks=checks, logs="\n".join(logs))

        rc, out = run_cmd(["npm", "install"], cwd=root, timeout_s=900)
        checks.append(f"npm_install_rc={rc}")
        logs.append(out)
        if rc != 0:
            return TestReport(ok=False, checks=checks, logs="\n".join(logs))

        rc, out = run_cmd(["npm", "run", "test:smoke"], cwd=root, timeout_s=300)
        checks.append(f"smoke_rc={rc}")
        logs.append(out)
        if rc != 0:
            return TestReport(ok=False, checks=checks, logs="\n".join(logs))

        rc, out = run_cmd(["npm", "run", "build"], cwd=root, timeout_s=900)
        checks.append(f"build_rc={rc}")
        logs.append(out)

        return TestReport(ok=(rc == 0), checks=checks, logs="\n".join(logs))
