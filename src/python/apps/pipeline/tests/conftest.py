"""Make this tests/ tree importable by a ``spawn`` child process (t49 A1).

Historical spawn-runtime tests submitted a module-level function to
a ``spawn`` ``ProcessPoolExecutor``. Pickle resolves a function by
``(module, qualname)``, and ``multiprocessing`` hands the child a copy of the
parent's ``sys.path`` — so the child can only unpickle the probe if this
directory's import root is on that path.

Under pytest's default ``prepend`` import mode it always was: pytest inserts the
rootdir of each collected package. t49 A1 switched the workspace gate to
``--import-mode=importlib`` — which is what lets six members each ship a
``tests/`` directory without competing for the top-level module name ``tests`` —
and importlib mode deliberately inserts nothing. The spawn child then dies with
``ModuleNotFoundError`` and the test fails as ``BrokenProcessPool``, which reads
like a resource problem and is not one.

This restores exactly the one path entry that was lost, and nothing else. It is
scoped to this directory rather than the workspace because this is the only
suite that pickles across a process boundary; a workspace-wide equivalent would
put six ``tests`` roots on ``sys.path`` and re-create the collision A1 removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).parent

# Both spellings, because the module name depends on whether `tests/__init__.py`
# exists: with it, the import root is `apps/pipeline` and the module is
# `tests.papers.…`; without it (the t49 A1 target state) the root is
# `apps/pipeline/tests` and the module is `papers.…`.
for _root in (_TESTS, _TESTS.parent):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
