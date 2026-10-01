"""Startup reconciliation for processes a previous run left behind.

Two stores in this tree outlive the process that wrote them, on purpose:
``data/containment_grants.json`` so a restart can reap rather than orphan, and
``data/bg_jobs.json`` so a restart never loses a detached job or its result.
Until now nothing read either of them at startup. A crashed or restarted server
therefore left every grant permanently "active" and every background job
permanently "running", and the first thing to touch one of those records was a
teardown aimed at a pid that had been reassigned in the meantime.

This module runs once, during startup, before anything of this run exists. That
timing is what makes its rules safe: every record it sees was written by an
earlier run, so "I cannot identify this process" is information about a previous
run's child and not about one of ours.

The two stores get **opposite** treatment, which is the whole reason this is a
module and not a loop:

* A **containment grant** is tied to a tool call that no longer has a caller.
  A live process under an abandoned grant is by definition an orphan, so it is
  torn down.
* A **background job** is detached deliberately and is documented to survive a
  uvicorn restart. Killing one here would break the feature, so its record is
  only corrected, never reaped. What gets fixed is identity: a job whose pid now
  belongs to someone else is retired so that nothing later signals the stranger.

Fail closed in both: a signal requires a positive identity from
:mod:`src.process_ownership`, and every other verdict is recorded rather than
acted on. Containment that cannot identify its target is not containment, and
the honest failure is a visible orphan rather than a dead bystander.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src import process_ownership

logger = logging.getLogger(__name__)


def reap_containment_grants() -> Dict[str, Any]:
    """Tear down or retire every grant a previous run left active.

    Per grant: a verified live process is torn down through
    :func:`src.containment.reap_record`; a grant whose process is gone is
    dropped; a grant naming a pid that is now someone else's is dropped
    *without a signal*, because the only thing left to do with it is stop
    believing it. A grant that cannot be verified at all is **kept**, so the
    orphan stays visible in ``active_grants()`` instead of being quietly
    written off as handled.
    """
    from src import containment

    report: Dict[str, Any] = {
        "seen": 0, "torn_down": 0, "already_gone": 0,
        "foreign": 0, "unverifiable": 0, "failed": 0,
    }
    try:
        records = containment.active_grants()
    except Exception:
        logger.warning("process_reaper: containment grant store unreadable", exc_info=True)
        return report

    for record in records:
        report["seen"] += 1
        grant_id = str(record.get("id") or "")
        if record.get("external"):
            # Nothing local ever ran, so there is nothing local to reap.
            containment.forget(grant_id)
            report["already_gone"] += 1
            continue
        verdict = process_ownership.verify_record(record)
        if verdict == process_ownership.GONE:
            if containment._group_present(record.get("pgid")):
                # Leader death does not prove tree death. Without a surviving
                # identity we cannot signal the group, so retain the evidence.
                report["failed"] += 1
                logger.error("process_reaper: grant %s leader is gone but group survives", grant_id)
                continue
            containment.forget(grant_id)
            report["already_gone"] += 1
            continue
        if verdict == process_ownership.FOREIGN:
            logger.warning(
                "process_reaper: grant %s named pid %s, which now belongs to a "
                "different process; dropping the record unsignalled",
                grant_id, record.get("pid"),
            )
            containment.forget(grant_id)
            report["foreign"] += 1
            continue
        if verdict == process_ownership.UNVERIFIABLE:
            logger.error(
                "process_reaper: grant %s (pid %s, owner %s) cannot be verified "
                "via %s; leaving it active and unsignalled — this is a "
                "containment failure, not a clean start",
                grant_id, record.get("pid"), record.get("owner"),
                process_ownership.inspection_mechanism(),
            )
            report["unverifiable"] += 1
            continue
        try:
            outcome = containment.reap_record(record)
        except Exception:
            logger.warning("process_reaper: tearing down grant %s failed", grant_id, exc_info=True)
            report["failed"] += 1
            continue
        if outcome.dead:
            containment.forget(grant_id)
            report["torn_down"] += 1
        else:
            logger.error(
                "process_reaper: grant %s survived teardown; survivors=%s",
                grant_id, list(outcome.survivors),
            )
            report["failed"] += 1
    return report


def reap_bg_jobs() -> Dict[str, Any]:
    """Correct the identity of background jobs a previous run launched.

    Deliberately kills nothing: a ``#!bg`` job is detached so that it outlives
    the request *and* the server, and the store exists so its result is still
    collected afterwards. The defect being closed is narrower — a record whose
    pid has been reassigned will be signalled by the max-runtime reaper an hour
    later, and that signal lands on whatever now holds the pid.
    """
    from src import bg_jobs

    try:
        return bg_jobs.disown_unverified()
    except Exception:
        logger.warning("process_reaper: background job store unreadable", exc_info=True)
        return {"seen": 0, "retired": 0, "kept": 0}


def reap_orphans() -> Dict[str, Any]:
    """Run both reconciliations. Returns a report; raises nothing.

    Blocking: a teardown escalates SIGTERM → grace → SIGKILL and waits for the
    process to actually go. Call it off the event loop.
    """
    report = {
        "mechanism": process_ownership.inspection_mechanism(),
        "grants": reap_containment_grants(),
        "bg_jobs": reap_bg_jobs(),
    }
    if report["mechanism"] == process_ownership.MECHANISM_NONE:
        logger.error(
            "process_reaper: this host offers no process inspection; no orphan "
            "from a previous run can be identified or reaped"
        )
    logger.info("process_reaper: startup reconciliation %s", report)
    return report


async def reap_orphans_at_startup() -> Dict[str, Any]:
    """:func:`reap_orphans` off the event loop, for an app startup task."""
    import asyncio

    return await asyncio.to_thread(reap_orphans)
