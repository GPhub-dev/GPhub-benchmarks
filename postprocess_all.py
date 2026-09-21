"""Regenerate the GPhub-kit plots and reports (results/img, results/report.log) of every benchmark case.

python postprocess_all.py
python postprocess_all.py --cases BM_01,Composite_2_100
python postprocess_all.py --jobs 4

"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPOSITE_DIMS = (
    2,
    4,
    8,
    16,
    32,
    48,
)
FORMATS = {"BM": ("png", "pdf"), "COMPOSITE": ("png",)}
SELECTION = ROOT.parent / "Scripts" / "benchmark_selection.csv"
SCRIPT_NAME = {
    "egobox-gp": "EGObox",
    "GaussianProcesses.jl": "GaussianProcessesjl",
    "scikit-learn": "scikitlearn",
}
LIBRARIES: tuple[str, ...] | None = None


def reported_libraries() -> tuple[str, ...]:
    """Script names of the libraries reported in the paper (column ``reported`` of the selection file)."""
    rows = [line.split(",") for line in SELECTION.read_text().splitlines()]
    head = rows[0]
    return tuple(
        SCRIPT_NAME.get(r[head.index("lib")], r[head.index("lib")])
        for r in rows[1:]
        if r[head.index("reported")] == "True"
    )


def cases(selection: list[str] | None) -> list[Path]:
    found = sorted(ROOT.glob("BM/BM_*")) + sorted(
        (
            p
            for p in ROOT.glob("COMPOSITE/Composite_*")
            if int(p.name.split("_")[1]) in COMPOSITE_DIMS
        ),
        key=lambda p: tuple(int(v) for v in p.name.split("_")[1:]),
    )
    found = [p for p in found if any((p / "results" / "storage").glob("lib_*.parquet"))]
    return [p for p in found if p.name in selection] if selection else found


def postprocess(case: Path) -> tuple[str, int, float, str]:
    group = case.parent.name
    img = case / "results" / "img"
    if img.exists():
        shutil.rmtree(img)
    code = f"import gphubkit; gphubkit.postprocess(formats={FORMATS[group]!r}, libraries={LIBRARIES!r})"
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=case,
        capture_output=True,
        text=True,
        check=False,
    )
    n_img = sum(1 for _ in img.rglob("*.p*")) if img.exists() else 0
    skipped = "; ".join(
        l.strip() for l in (out.stdout + out.stderr).splitlines() if "skipped" in l
    )
    if out.returncode != 0:
        skipped = (out.stderr or out.stdout).strip()[-600:]
    return (
        case.name,
        out.returncode,
        time.time() - t0,
        f"{n_img} files" + (f" | {skipped}" if skipped else ""),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cases", help="comma-separated case folders (default: all)")
    ap.add_argument(
        "--jobs", type=int, default=4, help="cases processed in parallel (default: 4)"
    )
    ap.add_argument(
        "--all-libraries",
        action="store_true",
        help="every stored prediction, not only the reported libraries",
    )
    a = ap.parse_args()
    global LIBRARIES
    LIBRARIES = (
        None if a.all_libraries or not SELECTION.exists() else reported_libraries()
    )
    print(
        "libraries:",
        "all stored predictions" if LIBRARIES is None else ", ".join(LIBRARIES),
        flush=True,
    )
    todo = cases(a.cases.split(",") if a.cases else None)
    failed = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for name, code, dt, info in pool.map(postprocess, todo):
            failed += code != 0
            print(
                f"{name:<20} {'ok' if code == 0 else 'FAILED':<7} {dt:7.1f} s  {info}",
                flush=True,
            )
    print(f"{len(todo)} cases, {failed} failed")
    from make_results_md import write_all

    n_cases, n_img = write_all([case.name for case in todo])
    print(f"RESULTS.md written for {n_cases} cases ({n_img} images)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
