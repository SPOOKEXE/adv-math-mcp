"""Where saved environments live.

The layout is owned by ``packages/workspace/src/layout.ts``; this is the smallest possible echo
of it, exactly as ``knowledge_tree.layout`` is. One directory name and the walk that finds a
project root — everything else about the convention stays on the TypeScript side, because two
full definitions of a path are a path that disagrees with itself.

The order is the whole design:

1. ``--root`` or ``OHMYPI_MATH_DIR``, because the harness knows which project it opened and an
   injected answer can never be wrong.
2. failing that, walk up for ``.ohmypi/`` or ``.git/`` and use ``.ohmypi/math``.

Saved environments are **not** cache. A scope is declarations, formulas and assumptions someone
built by hand over a session; losing it costs the work rather than a recomputation.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Mirrors ``OHMYPI_DIR`` in packages/workspace/src/layout.ts.
OHMYPI_DIR = ".ohmypi"

#: Mirrors ``SUBSYSTEMS.math``. Pinned by a conformance test on the TypeScript side.
MATH_DIR = "math"

#: Injection point. The harness sets this; a hand-started server usually does not.
ROOT_ENV = "OHMYPI_MATH_DIR"


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up for a project marker, nearest first.

    ``.ohmypi/`` outranks ``.git/`` at the same level: a directory that has one has declared
    itself a project, and a nested package with its own store must not be merged into the
    repository that happens to contain it.
    """
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / OHMYPI_DIR).is_dir():
            return candidate
        if (candidate / ".git").is_dir():
            return candidate

    return None


def resolve_root(explicit: str | None = None, start: Path | None = None) -> Path:
    """The environments directory, by injection where possible and discovery otherwise.

    Raises rather than guessing. A server that silently wrote to a directory under whatever
    happened to be the working directory would scatter saved environments across the filesystem,
    and the symptom is a `load` that reports a name it can plainly see was saved.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    from_env = os.environ.get(ROOT_ENV)
    if from_env:
        return Path(from_env).expanduser().resolve()

    root = find_project_root(start)
    if root is None:
        raise RuntimeError(
            f"no project root above {(start or Path.cwd()).resolve()}: "
            f"expected a {OHMYPI_DIR}/ or .git/ directory, or pass --root / set {ROOT_ENV}"
        )

    return root / OHMYPI_DIR / MATH_DIR
