"""Module-graph coverage for the split email library.

``specs/frontend.md`` records that this frontend has "no build-time type
checking, module graph validation, or script-order validation". That is
tolerable for a single 11k-line file and not tolerable for a package: splitting
``emailLibrary.js`` into modules that import each other adds three failure modes
nothing else here would catch.

1. The compatibility wrapper at ``static/js/emailLibrary.js`` drops an export.
   Five call sites import that path; a missing name is ``undefined`` at call
   time, not an error at load time, so the panel just stops responding.
2. A module in the package fails to evaluate — a stale relative specifier after
   a move, or a temporal-dead-zone read across an import cycle. The package has
   cycles by construction: extracted modules call back into ``index.js`` from
   event handlers. That is safe for hoisted function declarations and *not* safe
   for a ``const`` read while the graph is still evaluating, and which one you
   wrote is invisible in a diff.
3. A new module is missing from the ``sw.js`` precache, so the panel that works
   online cannot open offline.

The first and third are read off the source. The second needs a real module
loader, so it runs in a browser: each module is imported *on its own*, in a
fresh page, because entering the cycle at a submodule rather than at the entry
module is the order that exposes a dead-zone read.
"""

import json
import re
import subprocess
from pathlib import Path

from tests.helpers.js_modules import (
    EMAIL_LIBRARY_ENTRY,
    EMAIL_LIBRARY_PACKAGE,
    EMAIL_LIBRARY_WRAPPER,
    email_library_paths,
)

ROOT = Path(__file__).resolve().parents[1]
_SW = ROOT / "static" / "sw.js"

# `export function foo`, `export async function foo`, `export const foo`.
_EXPORT_DECL = re.compile(
    r"^export\s+(?:async\s+)?(?:function|const|let|class)\s+([A-Za-z_$][\w$]*)",
    re.M,
)
# `export { a, b } from '...'` and `export { a, b }`.
_EXPORT_LIST = re.compile(r"export\s*\{([^}]*)\}", re.S)


def _declared_exports(path: Path) -> set[str]:
    return set(_EXPORT_DECL.findall(path.read_text(encoding="utf-8")))


def _listed_exports(path: Path) -> set[str]:
    names: set[str] = set()
    for block in _EXPORT_LIST.findall(path.read_text(encoding="utf-8")):
        for raw in block.split(","):
            name = raw.strip().split(" as ")[-1].strip()
            if name:
                names.add(name)
    return names


# The email library's public surface. The entry module exports more than this —
# siblings in the package import helpers back out of it — so the wrapper is what
# declares which names are API and which are package-internal.
#
# Written out rather than derived because three of the five callers reach these
# through a dynamic import and a property read (`mod.openEmailLibrary` in
# chatStream.js and chatRenderer.js, `mod.refreshEmailLibrary` and
# `mod.openEmailLibrary` in document.js, `mod.mountEmailSettings` in
# settings.js), which no import scan can see. Only emailInbox.js imports names
# statically, and `test_wrapper_exposes_every_statically_imported_name` covers
# that half exactly.
_PUBLIC_SURFACE = {
    "closeEmailLibrary",
    "initEmailLibrary",
    "isOpen",
    "mountEmailSettings",
    "openEmailLibrary",
    "openEmailLibrarySettings",
    "prewarmEmailLibrary",
    "prewarmUnreadEmails",
    "refreshEmailLibrary",
}

_STATIC_JS = ROOT / "static" / "js"
_WRAPPER_IMPORT = re.compile(
    r"import\s*\{([^}]*)\}\s*from\s*'\./emailLibrary\.js(?:\?[^']*)?'", re.S
)


def test_wrapper_declares_the_public_surface():
    assert _listed_exports(EMAIL_LIBRARY_WRAPPER) == _PUBLIC_SURFACE


def test_wrapper_re_exports_only_names_the_entry_module_has():
    """A name in the wrapper that the entry module does not export is a
    SyntaxError at load time, and it takes the whole email panel with it."""
    entry = _declared_exports(EMAIL_LIBRARY_ENTRY) | _listed_exports(EMAIL_LIBRARY_ENTRY)
    wrapper = _listed_exports(EMAIL_LIBRARY_WRAPPER)
    assert wrapper, f"{EMAIL_LIBRARY_WRAPPER} re-exports nothing"
    assert wrapper <= entry, (
        "static/js/emailLibrary.js re-exports names static/js/emailLibrary/"
        f"index.js does not export: {sorted(wrapper - entry)}"
    )


def test_wrapper_exposes_every_statically_imported_name():
    """Whatever a module outside the package imports by name must be there."""
    wrapper = _listed_exports(EMAIL_LIBRARY_WRAPPER)
    checked = 0
    for path in sorted(_STATIC_JS.rglob("*.js")):
        if path == EMAIL_LIBRARY_WRAPPER or EMAIL_LIBRARY_PACKAGE in path.parents:
            continue
        for block in _WRAPPER_IMPORT.findall(path.read_text(encoding="utf-8")):
            for raw in block.split(","):
                name = raw.strip().split(" as ")[0].strip()
                if not name:
                    continue
                checked += 1
                assert name in wrapper, (
                    f"{path.relative_to(ROOT)} imports {name} from "
                    "static/js/emailLibrary.js, which does not export it"
                )
    assert checked, "no module imports names from static/js/emailLibrary.js"


def test_every_package_module_is_precached():
    sw = _SW.read_text(encoding="utf-8")
    for path in email_library_paths():
        url = "/static/js/emailLibrary/" + path.name
        assert f"'{url}'" in sw, (
            f"{url} is not in the static/sw.js precache, so a panel that works "
            "online will not open offline"
        )


def test_every_package_module_evaluates_on_its_own_in_a_browser():
    """Import each module first, alone, and require it to evaluate.

    Entering the package at a submodule is what turns an import cycle from
    harmless into a `ReferenceError: cannot access '…' before initialization`.
    Importing the entry module first would hide exactly that.
    """
    urls = ["/static/js/emailLibrary.js"] + [
        "/static/js/emailLibrary/" + p.name for p in email_library_paths()
    ]
    script = r'''
      import { chromium } from 'playwright';
      const origin = process.env.ODYSSEUS_TEST_STATIC_ORIGIN;
      const urls = JSON.parse(process.env.ODYSSEUS_EMAIL_MODULE_URLS);
      const browser = await chromium.launch({ headless: true });
      const results = {};
      for (const url of urls) {
        const page = await browser.newPage();
        // Same synthetic shell the other email browser tests use: these
        // modules touch #toast and #sidebar while evaluating.
        await page.goto(`${origin}/static/js/documentStats.js`);
        await page.setContent('<div id="toast"></div><div id="chat-container"></div><div id="sidebar"></div>');
        results[url] = await page.evaluate(async (target) => {
          try {
            const mod = await import(target);
            return { ok: true, exports: Object.keys(mod).sort() };
          } catch (err) {
            return { ok: false, error: String(err && err.message || err) };
          }
        }, url);
        await page.close();
      }
      console.log(JSON.stringify(results));
      await browser.close();
    '''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "ODYSSEUS_EMAIL_MODULE_URLS": json.dumps(urls),
        },
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    broken = {url: info["error"] for url, info in data.items() if not info["ok"]}
    assert not broken, f"modules that failed to evaluate on their own: {broken}"
    assert set(data) == set(urls)


def test_wrapper_and_entry_module_hand_out_the_same_functions():
    """Importing either path must give one live module instance.

    Two instances would mean two copies of the panel's module state, and the
    panel is a singleton keyed on DOM ids — the second copy would fight the
    first over `#email-lib-modal`.
    """
    script = r'''
      import { chromium } from 'playwright';
      const origin = process.env.ODYSSEUS_TEST_STATIC_ORIGIN;
      const browser = await chromium.launch({ headless: true });
      const page = await browser.newPage();
      await page.goto(`${origin}/static/js/documentStats.js`);
      await page.setContent('<div id="toast"></div><div id="chat-container"></div><div id="sidebar"></div>');
      const out = await page.evaluate(async () => {
        const wrapper = await import('/static/js/emailLibrary.js');
        const entry = await import('/static/js/emailLibrary/index.js');
        const names = Object.keys(wrapper).sort();
        return {
          names,
          identical: names.filter((n) => wrapper[n] === entry[n]),
          callable: names.filter((n) => typeof wrapper[n] === 'function'),
        };
      });
      console.log(JSON.stringify(out));
      await browser.close();
    '''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["names"], "the wrapper exported nothing at runtime"
    assert data["identical"] == data["names"], (
        "the wrapper and the entry module handed out different objects for "
        f"{sorted(set(data['names']) - set(data['identical']))}"
    )
    assert data["callable"] == data["names"], (
        "not every export is callable: "
        f"{sorted(set(data['names']) - set(data['callable']))}"
    )
