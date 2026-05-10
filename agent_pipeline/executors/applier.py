from __future__ import annotations

from pathlib import Path

from ..state import SiteState
from ..utils import safe_write


class SiteApplier:
    def materialize(self, state: SiteState) -> None:
        assert state.workdir is not None
        assert state.patchset is not None

        root = Path(state.workdir)

        # raw templates first
        for rel, html in state.raw_templates.items():
            safe_write(root / rel, html)

        # then apply generated artifacts
        for a in state.patchset.artifacts:
            safe_write(root / a.path, a.content)
