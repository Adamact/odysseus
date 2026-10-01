"""Shared test configuration - ensure project root is on sys.path and stub heavy deps."""
import sys
import os
import types
import importlib.util
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing core.database below runs init_db() at import time, and its default
# (sqlite:///./data/app.db) can't be opened in a clean worktree because SQLite
# won't create the missing ./data parent dir - pytest then dies during
# collection, before any test module loads. Default to an in-memory DB for the
# test session so collection is deterministic and writes no repo-local
# artifacts. An explicit DATABASE_URL (a real test/CI database) is preserved.
# This only unblocks collection/import-time init; it does not provide a shared
# file-backed DB across processes - tests needing that must set DATABASE_URL.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# Pre-import real heavy modules BEFORE any test file's module-level stubs can
# replace them with MagicMock. Some test files (e.g. test_llm_core_sanitize_*)
# stub sqlalchemy/core.database at module scope with `if mod not in sys.modules`,
# which fires during collection. If the real module hasn't been imported yet,
# the stub wins and contaminates every subsequent test that needs the real ORM.
try:
    import sqlalchemy  # noqa: F401
    import sqlalchemy.orm  # noqa: F401
    import core.database  # noqa: F401
    import src.database
except ImportError:
    pass  # not installed - the stubs below will handle it

def _has_module(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except (ImportError, ValueError):
        return False


# Stub optional dependencies only when they are not installed. Do not replace
# real FastAPI/Starlette/Pydantic modules: route tests import their subpackages.
for mod_name in [
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.types", "sqlalchemy.ext", "sqlalchemy.ext.declarative",
    "sqlalchemy.ext.hybrid", "sqlalchemy.sql", "sqlalchemy.sql.expression",
    "sqlalchemy.sql.sqltypes", "bcrypt", "pyotp",
    "httpx", "fastapi", "fastapi.responses", "fastapi.routing",
    "starlette", "starlette.responses", "starlette.middleware", "starlette.middleware.base",
    "pydantic",
]:
    if mod_name not in sys.modules and not _has_module(mod_name):
        sys.modules[mod_name] = MagicMock()

if "src.database" not in sys.modules:
    _db = types.ModuleType("src.database")
    _db.SessionLocal = MagicMock()
    _db.ModelEndpoint = MagicMock()
    sys.modules["src.database"] = _db

# Pre-import core.models before test_agent_loop.py's module-level stubs
# run (it replaces sys.modules['core.models'] with a MagicMock during
# collection, which breaks session import in subsequent tests).
import core.models  # noqa: E402

def pytest_configure(config):
    """Register the dynamic taxonomy ``sub_*`` markers before collection.

    The stable ``area_*`` markers are declared in ``pyproject.toml``. The
    per-file ``sub_*`` markers are derived from the test filenames here so that
    unknown-mark warnings still surface genuine typos outside the taxonomy. This
    only registers marker names; it imports no production module.
    """
    import pathlib
    from tests._taxonomy import discover_markers

    tests_dir = pathlib.Path(__file__).parent
    paths = list(tests_dir.rglob("test_*.py")) + list(tests_dir.rglob("*_test.py"))
    for marker_name in discover_markers(paths):
        if marker_name.startswith("sub_"):
            config.addinivalue_line("markers", f"{marker_name}: taxonomy sub-area marker")


def pytest_collection_modifyitems(config, items):
    """Tag each collected test with its taxonomy ``area_*`` and ``sub_*`` markers.

    Collection-time only: this adds markers and nothing else. It does not skip,
    reorder, or deselect tests, mutate fixtures or the environment, or import any
    production module. See ``tests/_taxonomy.py`` for the classification rules.
    """
    import pytest
    from tests._taxonomy import markers_for_path

    for item in items:
        path = getattr(item, "path", None) or item.fspath
        for marker_name in markers_for_path(path):
            item.add_marker(getattr(pytest.mark, marker_name))


@pytest.fixture(scope="session", autouse=True)
def _serve_test_static():
    """Serve static assets on loopback for the browser integration tests.

    Binds an ephemeral port so several worktrees can run their own suite at the
    same time, and publishes the resulting origin through
    ``ODYSSEUS_TEST_STATIC_ORIGIN``.  The browser tests shell out to node, which
    inherits the environment, so the snippets read the origin from
    ``process.env`` instead of hardcoding a port.

    Set ``ODYSSEUS_TEST_STATIC_PORT`` to pin a specific port when something
    outside pytest has to reach this server.
    """
    import os
    import threading
    import http.server
    import socketserver
    from pathlib import Path

    root_dir = Path(__file__).resolve().parent.parent

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root_dir), **kwargs)

        def log_message(self, format, *args):
            pass

        def guess_type(self, path):
            if path.endswith(".js") or path.endswith(".mjs"):
                return "application/javascript"
            if path.endswith(".css"):
                return "text/css"
            return super().guess_type(path)

    class _Server(socketserver.TCPServer):
        allow_reuse_address = True

    requested = int(os.environ.get("ODYSSEUS_TEST_STATIC_PORT") or 0)
    if not 0 <= requested <= 65535:
        raise ValueError("ODYSSEUS_TEST_STATIC_PORT must be between 0 and 65535")
    try:
        server = _Server(("127.0.0.1", requested), _Handler)
    except OSError as exc:
        # Port 0 cannot collide, so this only fires for an explicit pin.
        raise RuntimeError(
            f"ODYSSEUS_TEST_STATIC_PORT={requested} is not bindable; unset it to "
            "let the browser tests pick an ephemeral port"
        ) from exc

    origin = f"http://127.0.0.1:{server.server_address[1]}"
    previous_origin = os.environ.get("ODYSSEUS_TEST_STATIC_ORIGIN")
    os.environ["ODYSSEUS_TEST_STATIC_ORIGIN"] = origin

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield origin
    finally:
        if previous_origin is None:
            os.environ.pop("ODYSSEUS_TEST_STATIC_ORIGIN", None)
        else:
            os.environ["ODYSSEUS_TEST_STATIC_ORIGIN"] = previous_origin
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _no_leaked_module_stubs():
    """Fail the test that leaves a bare ``src.*``/``core.*`` stub behind.

    Several test modules install empty stand-in modules so an import-heavy
    production module can be loaded under the mocks above. When one of those
    writes is not undone, the stub stays in ``sys.modules`` for the rest of the
    session and every later test that imports the real module silently gets an
    empty one instead. The suite still passes as a whole, because the victims
    usually run before the leak; it only breaks under a different collection
    order, which is why this class of bug reaches CI green.

    This fixture is declared in the root conftest, so it is set up before any
    test-module fixture and torn down after all of them — a stub that a test's
    own teardown removes is not reported. The leaked entries are dropped here
    as well as reported, so the failure stays attributed to the test that
    introduced it instead of cascading into the rest of the run.

    Bare stubs present before the test starts are ignored: this guards against
    new leaks, it does not police import state the session began with.
    """
    from tests.helpers.import_state import bare_module_stubs, clear_module

    before = bare_module_stubs()
    yield
    leaked = sorted(bare_module_stubs() - before)
    if not leaked:
        return
    for name in leaked:
        clear_module(name)
    pytest.fail(
        "test left bare module stub(s) in sys.modules: "
        + ", ".join(leaked)
        + ". Register the stub through monkeypatch.setitem(sys.modules, ...) "
        "or tests.helpers.import_state.preserve_import_state so it is undone "
        "at teardown.",
        pytrace=False,
    )


@pytest.fixture(autouse=True)
def _no_context_window_network_probe(request):
    """Keep turn preparation from probing real provider metadata in tests.

    The compact runtime resolves its context window before the first model
    request. Tests that drive it with placeholder endpoints must not perform
    DNS or HTTP lookups; modules that exercise the probe opt in with a
    module-level ``CONTEXT_PROBE_NETWORK = True`` and supply their own client.
    """
    if getattr(request.module, "CONTEXT_PROBE_NETWORK", False):
        yield
        return
    try:
        from src.agent_runtime import context_resolution
    except Exception:
        yield
        return

    async def _disabled_probe(endpoint_url, model, headers, is_local, observations, errors, timeout):
        errors.append("probe_disabled_in_tests")

    # A private patcher keeps the shared ``monkeypatch`` fixture's teardown
    # order unchanged for tests that check their own sys.modules hygiene.
    patcher = pytest.MonkeyPatch()
    patcher.setattr(context_resolution, "_probe", _disabled_probe)
    context_resolution.clear_probe_cache()
    try:
        yield
    finally:
        patcher.undo()
        context_resolution.clear_probe_cache()
