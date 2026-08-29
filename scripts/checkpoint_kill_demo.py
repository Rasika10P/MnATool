"""Controller for the checkpoint-survives-a-kill demo (build order item 2, second half).

1. Runs checkpoint_start.py as a real subprocess. Watches its stdout for "[level]
   executing" -- proof that parse already completed and was checkpointed -- then sends a
   real SIGKILL, not a controlled shutdown.
2. Runs checkpoint_resume.py as a separate, fresh subprocess against the same thread_id and
   the same on-disk sqlite checkpoint file.
3. Asserts the resumed run's stdout never shows "[parse] executing" again (the checkpoint
   was actually used, not re-derived) and does show the run completing.

No settle buffer needed before the kill -- checkpoint_start.py and checkpoint_resume.py
both pass durability="sync" to .invoke(), which makes the write block until it's actually
on disk. LangGraph's default (durability="async") dispatches that write in the background,
and a hard kill can land before it completes; the batch fan-out demo
(scripts/batch_kill_demo.py) found this the hard way -- confirmed by direct inspection of
the checkpointer's writes table -- before this script needed a buffer to mask the same gap.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THREAD_ID = "kill-demo-thread"
DB_PATH = REPO_ROOT / "data" / "leveling_checkpoints.sqlite"


def run_and_capture(
    script_name: str, kill_after_marker: str | None, timeout: float = 60, extra_env: dict | None = None
) -> tuple[list[str], int | None, bool]:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script_name), THREAD_ID]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), **(extra_env or {})}
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines: list[str] = []
    killed = False
    start = time.time()
    for line in proc.stdout:
        print(f"  | {line}", end="")
        lines.append(line)
        if kill_after_marker and kill_after_marker in line:
            proc.send_signal(signal.SIGKILL)
            killed = True
            print(f"  [controller] SIGKILL sent immediately after seeing {kill_after_marker!r}")
            break
        if time.time() - start > timeout:
            proc.kill()
            print("  [controller] timed out waiting for marker -- killed as a fallback")
            break
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    return lines, proc.returncode, killed


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[controller] removed stale checkpoint db at {DB_PATH}")

    print(f"\n{'=' * 70}\nSTEP 1 -- start the run, kill it shortly after the level node begins\n{'=' * 70}")
    lines, returncode, killed = run_and_capture(
        "checkpoint_start.py", kill_after_marker="[level] executing",
        extra_env={"LEVELING_DEMO_DELAY_SECONDS": "8"},
    )

    assert killed, "controller never saw the kill marker -- demo invalid"
    assert any("[parse] executing" in l for l in lines), "parse never ran before the kill -- demo invalid"
    assert not any("COMPLETED" in l for l in lines), "run completed before the kill landed -- demo invalid, rerun"
    print(f"  process exit code: {returncode} (negative on Unix = killed by that signal number)")

    print(f"\n{'=' * 70}\nSTEP 2 -- resume from the same thread_id, in a brand-new process\n{'=' * 70}")
    lines2, returncode2, _ = run_and_capture("checkpoint_resume.py", kill_after_marker=None)

    parse_reran = any("[parse] executing" in l for l in lines2)
    level_ran = any("[level] executing" in l for l in lines2)
    completed = any("RESUMED RUN COMPLETED" in l for l in lines2)

    print(f"\n{'=' * 70}\nRESULT\n{'=' * 70}")
    print(f"parse re-executed on resume : {parse_reran}  (must be False)")
    print(f"level executed on resume    : {level_ran}  (must be True)")
    print(f"resumed run completed       : {completed}  (must be True)")

    assert not parse_reran, "PARSE NODE RE-RAN ON RESUME -- CHECKPOINTING IS BROKEN"
    assert level_ran and completed, "resume did not complete -- checkpointing is broken"
    print("\nCHECKPOINTING VERIFIED: the resumed run picked up after parse without re-running it.")


if __name__ == "__main__":
    main()
