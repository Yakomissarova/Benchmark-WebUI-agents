from __future__ import annotations

from pathlib import Path

from ..models import DockerReport
from ..utils import run_cmd, safe_write, dockerfile_nginx


class SiteDockerPackager:
    def build_image(self, root: Path, image_tag: str) -> DockerReport:
        rc, out = run_cmd(["docker", "version"], cwd=root, timeout_s=60)
        if rc != 0:
            return DockerReport(ok=False, image_tag=image_tag, build_logs=out)

        dockerfile = root / "Dockerfile"
        if not dockerfile.exists():
            safe_write(dockerfile, dockerfile_nginx())

        rc, out = run_cmd(["docker", "build", "-t", image_tag, "."], cwd=root, timeout_s=1800)
        return DockerReport(ok=(rc == 0), image_tag=image_tag, build_logs=out)
