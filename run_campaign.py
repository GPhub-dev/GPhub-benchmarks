"""Run GPhub-kit benchmark cases library by library.

python run_campaign.py --sync
python run_campaign.py --cases BM_07 --libs GPflow
python run_campaign.py --group COMPOSITE --langs py,m
python run_campaign.py --group BM --libs GPflow,tinygp

"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "campaign_log.jsonl"
COMPOSITE_DIMS = (2, 4, 8, 16, 32, 48)
COMPOSITE_SIZES = (100, 200, 500, 700, 1000)
EXT_LANG = {".py": "py", ".r": "r", ".jl": "jl", ".m": "m"}
PY_DIST = {
    "EGObox": "egobox",
    "GPJax": "gpjax",
    "GPy": "GPy",
    "GPyTorch": "gpytorch",
    "OpenTURNS": "openturns",
    "SMT": "smt",
    "scikitlearn": "scikit-learn",
    "GPflow": "gpflow",
    "tinygp": "tinygp",
    "UQpy": "UQpy",
}


MATLAB_VERSION = {
    "DACE": "2.5",
    "GPML": "4.2",
    "GPstuff": "4.7 (develop branch)",
    "STK": "2.8.1",
    "UQLab": "2.0",
}


def lib_name(script: Path) -> str:
    return script.stem.removeprefix("lib_").replace(
        "GaussianProcessesjl", "GaussianProcesses"
    )


def case_dirs(group: str | None, cases: list[str] | None) -> list[Path]:
    found = sorted(ROOT.glob("BM/BM_*")) + sorted(
        ROOT.glob("COMPOSITE/Composite_*"),
        key=lambda p: tuple(int(v) for v in p.name.split("_")[:0:-1]),
    )
    if cases:
        found = [p for p in found if p.name in cases]
    if group:
        found = [p for p in found if p.parent.name == group]
    return found


def sync_composite() -> None:
    """Copy the generic scripts into every COMPOSITE case."""
    import gphubkit

    database = Path(gphubkit.__file__).parent / "benchmark" / "database" / "COMPOSITE"
    for m in COMPOSITE_DIMS:
        for n in COMPOSITE_SIZES:
            case = ROOT / "COMPOSITE" / f"Composite_{m}_{n}"
            (case / "results" / "storage").mkdir(parents=True, exist_ok=True)
            scripts = case / "scripts"
            if scripts.exists():
                shutil.rmtree(scripts)
            shutil.copytree(
                ROOT / "_scripts",
                scripts,
                ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", "*.md"),
            )
            if not (case / "data" / "train_x.csv").exists():
                shutil.copytree(
                    database / f"Dim{m:02d}" / f"{n}" / "data",
                    case / "data",
                    dirs_exist_ok=True,
                )
    print("COMPOSITE cases synchronized with _scripts/")


def worker(case: Path, script: Path) -> None:
    import numpy as np
    import polars as pl
    from gphubkit.data import load
    from gphubkit.launcher import executor

    log_dir = case / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f"{script.stem}.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.captureWarnings(True)
    logger = logging.getLogger("gphubkit")

    train_x, test_x, train_y, _ = load(case / "data")
    run = {
        ".py": executor.python_executor,
        ".r": executor.r_executor,
        ".jl": executor.julia_executor,
        ".m": executor.matlab_executor,
    }[script.suffix]
    pred_y, pred_var, train_time, train_memory, pred_time, pred_memory = run(
        str(script), train_x, train_y, test_x, logger
    )
    pred_y, pred_var = (
        np.asarray(pred_y, dtype=float).flatten(),
        np.asarray(pred_var, dtype=float).flatten(),
    )
    if pred_y.size != test_x.shape[0]:
        msg = f"prediction of size {pred_y.size} for {test_x.shape[0]} test points"
        raise RuntimeError(msg)
    pl.DataFrame(
        {
            "pred_y": [pred_y],
            "pred_var": [pred_var],
            "train_time": [train_time],
            "train_memory": [train_memory],
            "pred_time": [pred_time],
            "pred_memory": [pred_memory],
        },
    ).write_parquet(case / "results" / "storage" / f"{script.stem}.parquet")
    print(
        json.dumps(
            {
                "train_time": train_time,
                "pred_time": pred_time,
                "train_memory": train_memory,
                "pred_memory": pred_memory,
                "finite": bool(np.all(np.isfinite(pred_y))),
            }
        )
    )


def version_of(lib: str, lang: str) -> str:
    try:
        if lang == "py":
            from importlib.metadata import version

            return version(PY_DIST[lib])
        if lang == "r":
            out = subprocess.run(
                ["Rscript", "-e", f'cat(as.character(packageVersion("{lib}")))'],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return f"{out.stdout.strip()} (R {r_version()})"
        if lang == "jl":
            return julia_versions().get(lib, "?")
        if lang == "m":
            return f"{MATLAB_VERSION.get(lib, '?')} (MATLAB R2026a)"
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"
    return "?"


def r_version() -> str:
    out = subprocess.run(
        ["Rscript", "-e", 'cat(paste(R.version$major, R.version$minor, sep="."))'],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return out.stdout.strip()


_JL: dict[str, str] = {}


def julia_versions() -> dict[str, str]:
    if not _JL:
        code = (
            "import juliacall as jl; print(jl.Main.seval('import Pkg; v=[p.version for p in values(Pkg.dependencies()) "
            'if p.name=="GaussianProcesses"][1]; string(v, " (Julia ", VERSION, ")")\'))'
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        _JL["GaussianProcesses"] = (
            out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "?"
        )
    return _JL


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help="copy _scripts/ (and missing data) into the COMPOSITE cases",
    )
    ap.add_argument("--group", choices=["BM", "COMPOSITE"])
    ap.add_argument(
        "--cases", help="comma-separated case folders, e.g. BM_07,Composite_16_100"
    )
    ap.add_argument(
        "--libs",
        help="comma-separated libraries, e.g. GPflow,SMT (default: all scripts)",
    )
    ap.add_argument(
        "--langs", default="py,r,jl,m", help="languages to run (default: py,r,jl,m)"
    )
    ap.add_argument(
        "--timeout", type=float, default=3600.0, help="seconds per (case, library)"
    )
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help="skip pairs already logged as ok in this campaign",
    )
    ap.add_argument("--tag", default="campaign-2", help="label written to the log")
    ap.add_argument(
        "--worker", nargs=2, metavar=("CASE", "SCRIPT"), help=argparse.SUPPRESS
    )
    a = ap.parse_args()

    if a.worker:
        worker(Path(a.worker[0]), Path(a.worker[1]))
        return
    if a.sync:
        sync_composite()
    if not (a.group or a.cases):
        return

    done = set()
    if a.skip_done and LOG.exists():
        for line in LOG.read_text().splitlines():
            rec = json.loads(line)
            if rec["status"] == "ok" and rec["tag"] == a.tag:
                done.add((rec["case"], rec["lib"]))
    libs = a.libs.split(",") if a.libs else None
    langs = a.langs.split(",")
    timed_out: dict[str, list[tuple[int, int]]] = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            rec = json.loads(line)
            if (
                rec["status"] == "timeout"
                and rec["tag"] == a.tag
                and rec["case"].startswith("Composite_")
            ):
                m_, n_ = (int(v) for v in rec["case"].split("_")[1:])
                timed_out.setdefault(rec["lib"], []).append((m_, n_))
    for case in case_dirs(a.group, a.cases.split(",") if a.cases else None):
        if not (case / "data" / "train_x.csv").exists():
            continue
        for script in sorted((case / "scripts").glob("lib_*")):
            lib, lang = lib_name(script), EXT_LANG.get(script.suffix)
            if (
                lang not in langs
                or (libs and lib not in libs)
                or (case.name, lib) in done
            ):
                continue
            stored = case / "results" / "storage" / f"{script.stem}.parquet"
            stored.unlink(missing_ok=True)
            t0 = time.time()
            rec = {
                "tag": a.tag,
                "date": dt.datetime.now().isoformat(timespec="seconds"),
                "case": case.name,
                "lib": lib,
                "lang": lang,
                "version": version_of(lib, lang),
            }
            if case.name.startswith("Composite_"):
                m_, n_ = (int(v) for v in case.name.split("_")[1:])
                smaller = [
                    (m0, n0)
                    for m0, n0 in timed_out.get(lib, [])
                    if m0 <= m_ and n0 <= n_
                ]
                if smaller:
                    rec.update(
                        status="skipped",
                        error=f"timeout for m={smaller[0][0]}, n={smaller[0][1]}",
                        wall=0.0,
                    )
                    with LOG.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(
                        f"{rec['date']}  {case.name:<20} {lib:<18} skipped  ({rec['error']})",
                        flush=True,
                    )
                    continue
            try:
                out = subprocess.run(
                    [sys.executable, __file__, "--worker", str(case), str(script)],
                    capture_output=True,
                    text=True,
                    timeout=a.timeout,
                    check=False,
                    cwd=case,
                )
                if out.returncode == 0 and stored.exists():
                    rec.update(
                        status="ok", **json.loads(out.stdout.strip().splitlines()[-1])
                    )
                else:
                    rec.update(
                        status="failed",
                        error=(out.stderr or out.stdout).strip()[-1500:],
                    )
            except subprocess.TimeoutExpired:
                rec.update(
                    status="timeout", error=f"no result within {a.timeout:.0f} s"
                )
                if case.name.startswith("Composite_"):
                    timed_out.setdefault(lib, []).append((m_, n_))
            rec["wall"] = round(time.time() - t0, 2)
            with LOG.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"{rec['date']}  {case.name:<20} {lib:<18} {rec['status']:<8} wall {rec['wall']:>8.1f} s  "
                f"train {rec.get('train_time', float('nan')):>9.3f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
