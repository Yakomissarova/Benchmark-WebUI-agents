from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_cmd(cmd: List[str], cwd: Path, timeout_s: int = 900) -> Tuple[int, str]:
    """
    Synchronous command runner (as in many monolithic scripts).
    Returns (return_code, combined_stdout_stderr).
    """
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        timeout=timeout_s,
        capture_output=True,
        text=True,
        check=False,
    )
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out


def guess_entry_html(raw_templates: Dict[str, str]) -> str:
    keys = list(raw_templates.keys())
    for k in keys:
        if k.lower().endswith("index.html"):
            return k
    return keys[0] if keys else "index.html"


def normalize_rel_paths(html: str) -> str:
    # conservative normalization
    html = html.replace('href="/', 'href="./')
    html = html.replace('src="/', 'src="./')
    return html


def default_package_json(app_name: str) -> str:
    pkg = {
        "name": app_name,
        "private": True,
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "preview": "vite preview --host",
            "test:smoke": "node scripts/smoke.mjs",
        },
        "devDependencies": {"vite": "^5.0.0"},
    }
    return json.dumps(pkg, indent=2)


def default_vite_config() -> str:
    return """import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 5173, strictPort: true },
  build: { outDir: 'dist' },
});
"""


def default_smoke_script(entry_html: str) -> str:
    entry_name = Path(entry_html).name
    return f"""import fs from 'node:fs';
import path from 'node:path';

const dist = path.resolve(process.cwd(), 'dist');
if (!fs.existsSync(dist)) {{
  console.error('dist folder not found');
  process.exit(1);
}}

const entry = path.resolve(dist, '{entry_name}');
if (!fs.existsSync(entry)) {{
  console.error('entry html not found in dist:', entry);
  process.exit(1);
}}

console.log('smoke ok');
"""


def dockerfile_nginx() -> str:
    return """# syntax=docker/dockerfile:1
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci || npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
"""
