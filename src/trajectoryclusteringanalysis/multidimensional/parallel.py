"""Parallel execution of Optuna trials across processes (one trial per process)."""
import os
import sys


def _run_trial(objective, params, sys_path, torch_threads):
    # Executed in a worker process: restore the import path and limit torch threads.
    for p in reversed(sys_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    from optuna.trial import FixedTrial

    torch.set_num_threads(torch_threads)
    trial = FixedTrial(params)
    try:
        values = objective(trial)
    except Exception as exc:
        return None, {}, repr(exc)
    return values, dict(trial.user_attrs), None


def _call(func, args, sys_path, torch_threads, log_path=None):
    for p in reversed(sys_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch

    torch.set_num_threads(torch_threads)
    if log_path is None:
        return func(*args)
    # Prints and progress bars of this task go to its own log file
    import contextlib

    with open(log_path, "w", buffering=1, encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            return func(*args)


def _last_line(path, size=4000):
    # Last non-empty line of a log (progress bars update with carriage returns)
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - size))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    parts = [s.strip() for s in text.replace("\r", "\n").split("\n") if s.strip()]
    return parts[-1][:110] if parts else ""


def _status_html(futures, logs):
    # Title and one bullet per task: waiting, running (last log line) or done
    from html import escape
    from IPython.display import HTML

    items = []
    for i, f in enumerate(futures):
        if f.done():
            text = "terminé"
        elif f.running():
            text = escape(_last_line(logs[i])) or "démarrage…"
        else:
            text = "en attente"
        items.append(f"<li><b>Essai {i}</b> : <code>{text}</code></li>")
    return HTML("<b>Avancement des essais</b><ul>" + "".join(items) + "</ul>")


def parallel_map(func, args_list, n_jobs=None, torch_threads=2, poll_s=3.0):
    """
    Evaluates `func(*args)` for each tuple in `args_list` in separate processes
    and returns the results in the same order. `func` may be defined in a notebook.

    A tqdm bar advances as each task completes, and the last log line of each
    running task (e.g. the Lightning epoch bar) is shown below it, refreshed
    every `poll_s` seconds. Full logs are kept in a temporary folder if a task fails.
    """
    import shutil
    import tempfile
    from concurrent.futures import FIRST_COMPLETED, wait
    from joblib.externals.loky import get_reusable_executor
    from tqdm.auto import tqdm

    try:
        from IPython.display import HTML, display
        handle = display(HTML(""), display_id=True)
    except Exception:
        handle = None

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) // max(1, torch_threads))
    n_jobs = max(1, min(int(n_jobs), len(args_list)))
    sys_path = list(sys.path)
    log_dir = tempfile.mkdtemp(prefix="tca_parallel_")
    logs = [os.path.join(log_dir, f"task_{i}.log") for i in range(len(args_list))]
    executor = get_reusable_executor(max_workers=n_jobs)
    futures = [
        executor.submit(_call, func, args, sys_path, torch_threads, logs[i])
        for i, args in enumerate(args_list)
    ]
    index = {f: i for i, f in enumerate(futures)}
    results = [None] * len(futures)
    pending = set(futures)
    with tqdm(total=len(futures), desc="Trials") as pbar:
        while pending:
            done, pending = wait(pending, timeout=poll_s, return_when=FIRST_COMPLETED)
            for f in done:
                results[index[f]] = f.result()
                pbar.update(1)
            if handle is not None:
                handle.update(_status_html(futures, logs))
    if handle is not None:
        handle.update(HTML(""))
    shutil.rmtree(log_dir, ignore_errors=True)
    return results


def optimize_parallel(study, objective, search_space, n_trials, n_jobs=None, torch_threads=2):
    """
    Runs `n_trials` of `objective` with `n_jobs` trials evaluated concurrently in
    separate processes, using Optuna's ask/tell interface.

    Parameters are sampled in the main process from `search_space`, then the
    objective is replayed in a worker with a `FixedTrial`, so the objective can
    keep using `trial.suggest_*` unchanged.

    Trials are sampled in batches of `n_jobs`: the sampler learns from the
    previous batch only.

    Parameters:
    - study: an optuna Study
    - objective: function(trial) -> float or tuple of floats; attributes set with
      `trial.set_user_attr` are propagated back to the study.
    - search_space: dict name -> optuna distribution, matching the `suggest_*`
      calls of the objective (e.g. IntDistribution(2, 5)).
    - n_trials: total number of trials
    - n_jobs: number of concurrent processes (default: CPUs // torch_threads)
    - torch_threads: torch intra-op threads per process
    """
    import optuna
    from joblib import Parallel, delayed

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) // max(1, torch_threads))
    n_jobs = max(1, min(int(n_jobs), int(n_trials)))

    sys_path = list(sys.path)
    remaining = int(n_trials)
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        while remaining > 0:
            batch = min(n_jobs, remaining)
            trials = [study.ask(fixed_distributions=search_space) for _ in range(batch)]
            outputs = parallel(
                delayed(_run_trial)(objective, t.params, sys_path, torch_threads)
                for t in trials
            )
            for trial, (values, attrs, error) in zip(trials, outputs):
                for k, v in attrs.items():
                    trial.set_user_attr(k, v)
                if error is not None:
                    print(f"Trial {trial.number} failed: {error}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial, values)
            remaining -= batch
