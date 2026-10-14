"""E3 background worker: runs the Rosetta authors' own code, unchanged, for every pending task.

Task kinds
  align     experiments.alignment.run_alignment(AlignmentRun(dataset, vision, language, seed, "ours",
            classification=())) -> FOSCTTM on all 40,504 MS COCO val pairs. Classification is skipped: it is evaluated
            after retrieval with fresh generators, so FOSCTTM is identical to the paper's pipeline.
  geometry  experiments.shared_geometry.run(GeometryRun(dataset, vision, language)) -> their CKA (unbiased, fp64),
            mutual k-NN, TSI and QSI on 10,000 paired training items, seed 42 (Fig. 3c protocol).

Usage: python e3_worker.py <tasks.json> <out_dir> <rosetta_repo> <restart_workers> [log file name]
"""
import json, os, sys, time, traceback, types


def _setup(repo):
    sys.path.insert(0, repo)
    # The Rosetta code targets Python 3.14 (lazy annotations). On older Pythons one annotation in
    # unpaired_rosetta/results.py names `pd`, imported only for type checkers; make it resolvable.
    # This affects a type hint only, never a computation.
    if sys.version_info < (3, 14):
        import builtins, pandas
        builtins.pd = pandas
    # experiments.alignment imports experiments.methods, which imports every baseline (and their extra
    # dependencies). Only "ours" is needed, so register a minimal methods module that builds the authors'
    # WassersteinProcrustes exactly as experiments.methods.build_aligner("ours", ...) does.
    from unpaired_rosetta.wasserstein_procrustes import WassersteinProcrustes
    stub = types.ModuleType("experiments.methods")

    def build_aligner(method, num_workers=1, num_threads=1):
        assert method == "ours", method
        return WassersteinProcrustes(num_workers=num_workers, num_threads=num_threads)

    stub.build_aligner = build_aligner
    import experiments  # noqa: F401  (package)
    sys.modules["experiments.methods"] = stub


def _task_path(out_dir, t):
    if t["kind"] == "align":
        return os.path.join(out_dir, "align", f"{t['dataset']}__{t['vision']}__{t['language']}__seed{t['seed']}.json")
    return os.path.join(out_dir, "geometry", f"{t['dataset']}__{t['vision']}__{t['language']}.json")


def run_task(t, out_dir, repo, restart_workers=1):
    _setup(repo)
    path = _task_path(out_dir, t)
    if os.path.exists(path):
        return path, "exists", 0.0
    start = time.time()
    if t["kind"] == "align":
        from experiments.alignment import AlignmentRun, run_alignment
        cfg = dict(dataset=t["dataset"], model_x=t["vision"], model_y=t["language"], seed=t["seed"],
                   method="ours", classification=())
        for k in ("train_size", "validation_size"):          # only set in the local dry run
            if t.get(k) is not None:
                cfg[k] = t[k]
        # The 30 initialization restarts (each a 30-node QAP solved by MPOpt) run in `restart_workers` spawned
        # processes. The authors guarantee this gives exactly the same map as a sequential run; it only changes speed.
        _, metrics = run_alignment(AlignmentRun(**cfg), num_workers=restart_workers, num_threads=min(4, restart_workers))
    else:
        import experiments.shared_geometry as S
        cfg = dict(dataset=t["dataset"], model_x=t["vision"], model_y=t["language"])
        if t.get("num_samples") is not None:
            cfg["num_samples"] = t["num_samples"]
        metrics = S.run(S.GeometryRun(**cfg))
    rec = dict(task=t, metrics=metrics, seconds=time.time() - start)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w")); os.replace(tmp, path)      # atomic: a crash never leaves a half file
    return path, "done", rec["seconds"]


if __name__ == "__main__":
    tasks_file, out_dir, repo = sys.argv[1], sys.argv[2], sys.argv[3]
    restart_workers = int(sys.argv[4])
    log_name = sys.argv[5] if len(sys.argv) > 5 else "worker_log.txt"
    tasks = json.load(open(tasks_file))
    pending = [t for t in tasks if not os.path.exists(_task_path(out_dir, t))]
    log = open(os.path.join(out_dir, log_name), "a")
    log.write(f"{time.strftime('%X')} start: {len(pending)} pending of {len(tasks)} tasks; one task at a time, "
              f"{restart_workers} processes for the initialization restarts\n"); log.flush()
    # One task at a time in this process; all cores go to the restarts of the current alignment.
    for t in pending:
        log.write(f"{time.strftime('%X')} begin {os.path.basename(_task_path(out_dir, t))}\n"); log.flush()
        try:
            path, status, secs = run_task(t, out_dir, repo, restart_workers)
            log.write(f"{time.strftime('%X')} {status} {os.path.basename(path)} ({secs:.0f}s)\n")
        except Exception:
            log.write(f"{time.strftime('%X')} FAILED {t}\n{traceback.format_exc()}\n")
        log.flush()
    log.write(f"{time.strftime('%X')} finished\n"); log.flush()
