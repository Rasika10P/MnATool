"""Controller for the real batch kill-mid-fan-out demo (build order item 3).

Staggers each of the 25 employees' start (LEVELING_DEMO_STAGGER_SECONDS) so a subset
reliably completes before a real SIGKILL lands, then resumes in a fresh process and checks
that no employee who completed before the kill runs again.

Reads subprocess stdout on a background thread rather than the main thread's blocking
`for line in proc.stdout` loop -- the first version of this controller blocked on that read
during any quiet stretch, which meant the wall-clock kill trigger never got a chance to
fire until the next line arrived. Timing the kill off a fixed sleep on the main thread,
independent of subprocess output, is what actually lands it at the intended point.

batch_start.py and batch_resume.py both pass durability="sync" to .invoke() -- without it,
a completed employee's checkpoint write can still be in flight (LangGraph's default,
durability="async", dispatches it in the background) when the kill lands, and that
employee gets silently re-run on resume. Confirmed by direct inspection of the
checkpointer's writes table: even settling several seconds after a "done" print, with
durability="async" no individual task write ever showed up while sibling tasks were still
running -- only durability="sync" fixed it. No settle buffer is needed once that's set.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THREAD_ID = "batch-kill-demo-thread"
DB_PATH = REPO_ROOT / "data" / "leveling_batch_checkpoints.sqlite"
STAGGER_SECONDS = 1.0
# batch_start.py defaults --limit to 3 (cost control), but this demo's whole point is a
# real mid-batch kill -- a population of 3 would finish well inside the stagger window and
# never demonstrate partial completion, so it explicitly overrides the default.
POPULATION_LIMIT = 25
KILL_AT_SECONDS = 20.0  # real per-call latency has ranged ~1-13s this session on top of
                         # the stagger delay, so 12s left zero completions the first try;
                         # 20s gives the low-index employees room to actually finish while
                         # the high-index ones (stagger 20s+) haven't started their real
                         # call yet


def read_lines(proc: subprocess.Popen, sink: list[str], t0: float) -> None:
    for line in proc.stdout:
        sink.append(line)
        print(f"  t={time.time() - t0:5.2f}s | {line}", end="")


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[controller] removed stale checkpoint db at {DB_PATH}")

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "LEVELING_DEMO_STAGGER_SECONDS": str(STAGGER_SECONDS)}

    print(f"\n{'=' * 70}\nSTEP 1 -- start the real {POPULATION_LIMIT}-employee batch, staggered {STAGGER_SECONDS}s apart\n{'=' * 70}")
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "batch_start.py"), THREAD_ID, "--limit", str(POPULATION_LIMIT)],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines1: list[str] = []
    t0 = time.time()
    reader = threading.Thread(target=read_lines, args=(proc, lines1, t0), daemon=True)
    reader.start()

    time.sleep(KILL_AT_SECONDS)
    proc.send_signal(signal.SIGKILL)
    print(f"  [controller] SIGKILL sent at t={time.time() - t0:.2f}s")
    proc.wait(timeout=15)
    reader.join(timeout=5)

    done_in_run1 = {l.split("done for ")[1].strip() for l in lines1 if "done for" in l}
    assert done_in_run1, "nothing completed before the kill -- demo invalid, raise KILL_AT_SECONDS"
    assert len(done_in_run1) < POPULATION_LIMIT, "everything completed before the kill -- demo invalid, lower KILL_AT_SECONDS"
    print(f"  completed before kill: {len(done_in_run1)}/{POPULATION_LIMIT} -> {sorted(done_in_run1)}")

    print(f"\n{'=' * 70}\nSTEP 2 -- resume from the same thread_id, in a brand-new process\n{'=' * 70}")
    proc2 = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "batch_resume.py"), THREAD_ID],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines2: list[str] = []
    for line in proc2.stdout:
        print(f"  | {line}", end="")
        lines2.append(line)
    proc2.wait(timeout=120)

    done_in_run2 = {l.split("done for ")[1].strip() for l in lines2 if "done for" in l}
    completed = any("RESUMED BATCH COMPLETED" in l for l in lines2)
    overlap = done_in_run1 & done_in_run2

    print(f"\n{'=' * 70}\nRESULT\n{'=' * 70}")
    print(f"completed before kill : {len(done_in_run1)}/{POPULATION_LIMIT}")
    print(f"re-run during resume  : {len(done_in_run2)}/{POPULATION_LIMIT - len(done_in_run1)} remaining")
    print(f"re-executed (overlap) : {sorted(overlap)}  (must be empty)")
    print(f"resumed run completed : {completed}  (must be True)")

    assert not overlap, f"EMPLOYEES RE-EXECUTED ON RESUME -- CHECKPOINTING IS BROKEN: {overlap}"
    assert completed, "resume did not complete"
    print("\nCHECKPOINTING VERIFIED: employees completed before the kill were not re-run on resume.")


if __name__ == "__main__":
    main()
