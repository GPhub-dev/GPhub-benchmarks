"""Write a RESULTS.md gallery in every benchmark case, and an index at the root of the repository.

python make_results_md.py
python make_results_md.py --cases BM_02,Composite_8_500

"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPOSITE_DIMS = (2, 4, 8, 16, 32, 48)
COMPOSITE_SIZES = (100, 200, 500, 700, 1000)
BM_CASES = {
    "BM_01": (1, "Damped cosine", "Gaussian"),
    "BM_02": (2, "Branin-Hoo", "Matérn 5/2"),
    "BM_04": (3, "Dette & Pepelyshev", "Gaussian"),
    "BM_05": (4, "OTL circuit", "Gaussian"),
    "BM_06": (5, "Borehole", "Matérn 5/2"),
    "BM_07": (6, "Morris", "Matérn 5/2"),
}
PHYSICAL_UNITS = {
    "BM_06",
    "BM_07",
}
DISPLAY_NAME = {
    "EGObox": "egobox-gp",
    "GaussianProcessesjl": "GaussianProcesses.jl",
    "scikitlearn": "scikit-learn",
}
RADAR = {
    "R2": "R²",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "MSE": "MSE",
    "MedAE": "MedAE",
    "NLPD": "NLPD",
    "MSLL": "MSLL",
    "train_time": "Training time",
    "pred_time": "Prediction time",
    "train_memory": "Peak memory, training",
    "pred_memory": "Peak memory, prediction",
}
PLOTS = {
    "prediction": (
        "Prediction",
        "True function and predictive mean, with the training points (inputs of dimension one or two).",
        2,
        440,
    ),
    "crossvalidation": (
        "Actual versus predicted values",
        "Test set: actual versus predicted values (left) and residuals (right).",
        2,
        440,
    ),
    "density": (
        "Distribution of the outputs",
        "Test set: distributions of the actual outputs and of the predictions.",
        4,
        215,
    ),
    "residuals": (
        "Distribution of the residuals",
        "Test set: distribution of the residuals.",
        4,
        215,
    ),
}


def case_dirs(selection: list[str] | None = None) -> list[Path]:
    found = sorted(ROOT.glob("BM/BM_*")) + sorted(
        (
            p
            for p in ROOT.glob("COMPOSITE/Composite_*")
            if int(p.name.split("_")[1]) in COMPOSITE_DIMS
        ),
        key=lambda p: tuple(int(v) for v in p.name.split("_")[1:]),
    )
    found = [p for p in found if (p / "results" / "img").is_dir()]
    return [p for p in found if p.name in selection] if selection else found


def _shape(csv: Path) -> tuple[int, int]:
    """Rows and columns of a headerless CSV file."""
    with csv.open() as f:
        first = f.readline()
        rows = 1 + sum(1 for _ in f)
    return rows, first.count(",") + 1


def describe(case: Path) -> tuple[str, str]:
    """Title and one-line description of a case."""
    n, d = _shape(case / "data" / "train_x.csv")
    n_test, _ = _shape(case / "data" / "test_x.csv")
    if case.name in BM_CASES:
        k, name, kernel = BM_CASES[case.name]
        return (
            f"{case.name} · Analytical test function {k}: {name}",
            f"Input dimension d = {d}, n = {n} training points, {n_test} test points, {kernel} kernel, constant trend. "
            "Native optimizer settings of each library.",
        )
    return (
        f"{case.name} · Composite plate, m = {d}, n = {n}",
        f"m = {d} ply angles, n = {n} training points, {n_test} test points, Gaussian kernel, constant trend. "
        "Harmonized specification (see [`_scripts/README.md`](../../_scripts/README.md)).",
    )


def lib_of(png: Path) -> str:
    stem = png.stem
    for prefix in ("lib_", "residuals_", "density_", "prediction_"):
        stem = stem.removeprefix(prefix)
    return DISPLAY_NAME.get(stem, stem)


def metrics_table(case: Path) -> list[str]:
    """Metrics table of results/report.log as Markdown (the report itself is not under version control)."""
    report = case / "results" / "report.log"
    if not report.exists():
        return []
    rows = []
    for line in report.read_text().splitlines():
        if line.startswith(("┃", "│")):
            rows.append(
                [c.strip() for c in line.strip("┃│").replace("┃", "│").split("│")]
            )
    if len(rows) < 2:
        return []
    out = [
        "| " + " | ".join(f"**{c}**" for c in rows[0]) + " |",
        "|:--|" + "--:|" * (len(rows[0]) - 1),
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


def _cell(case: Path, png: Path, caption: str, width: int) -> str:
    """One gallery cell: the PNG shown in place (GitHub renders PNG, not PDF); a click opens it at full size."""
    rel = png.relative_to(case).as_posix()
    return (
        f'<td align="center"><a href="{rel}"><img src="{rel}" alt="{caption}" width="{width}"></a>'
        f"<br><sub>{caption}</sub></td>"
    )


def gallery(
    case: Path, items: list[tuple[Path, str]], columns: int, width: int
) -> list[str]:
    out = ["<table>"]
    for i in range(0, len(items), columns):
        out.append(
            "<tr>"
            + "".join(_cell(case, p, c, width) for p, c in items[i : i + columns])
            + "</tr>"
        )
    return [*out, "</table>", ""]


def write_case(case: Path) -> int:
    img = case / "results" / "img"
    title, text = describe(case)
    libs = sorted(
        {lib_of(p) for p in (img / "crossvalidation").glob("*.png")}, key=str.lower
    )
    lines = [
        f"# {title}",
        "",
        text,
        "",
        f"Libraries ({len(libs)}): " + ", ".join(f"`{l}`" for l in libs) + ".",
        "",
        "Generated by `make_results_md.py` from `results/img` and `results/report.log`; do not edit by hand. "
        "Click an image to open it at full size.",
        "",
    ]
    n_img = 0
    table = metrics_table(case)
    if table:
        units = (
            "output in physical units"
            if case.name in PHYSICAL_UNITS
            else "standardized output"
        )
        lines += [
            "## Metrics",
            "",
            f"Errors on the test set ({units}); NLPD and MSLL per test point.",
            "",
            *table,
            "",
        ]
    overview = img / "radar_by_library.png"
    radars = {
        p.stem.removeprefix("radar_"): p
        for p in sorted(img.glob("radar_*.png"))
        if p != overview
    }
    if overview.exists() or radars:
        lines += ["## Comparison of the libraries", ""]
    if overview.exists():
        lines += [
            "All metrics at once, each divided by its largest value over the libraries "
            "(absolute values for the NLPD and the MSLL).",
            "",
        ]
        lines += gallery(case, [(overview, "All metrics, by library")], 1, 760)
        n_img += 1
    if radars:
        order = [k for k in RADAR if k in radars] + sorted(
            k for k in radars if k not in RADAR
        )
        lines += ["One radar chart per metric.", ""]
        lines += gallery(case, [(radars[k], RADAR.get(k, k)) for k in order], 3, 290)
        n_img += len(order)
    # plots of each library
    folders = [f for f in PLOTS if (img / f).is_dir()] + sorted(
        p.name for p in img.iterdir() if p.is_dir() and p.name not in PLOTS
    )
    for folder in folders:
        pngs = sorted((img / folder).glob("*.png"), key=lambda p: lib_of(p).lower())
        if not pngs:
            continue
        name, note, columns, width = PLOTS.get(
            folder, (folder.capitalize(), "", 3, 290)
        )
        lines += [f"## {name}", "", note, ""] if note else [f"## {name}", ""]
        lines += gallery(case, [(p, lib_of(p)) for p in pngs], columns, width)
        n_img += len(pngs)
    (case / "RESULTS.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    return n_img


def write_index(cases: list[Path]) -> None:
    names = {c.name for c in cases}
    lines = [
        "# Results",
        "",
        "One page per case, with the metrics table and every plot of the toolkit. "
        "Generated by `make_results_md.py`; do not edit by hand.",
        "",
        "## Analytical test functions",
        "",
        "| Function | Case | Kernel | Results |",
        "|:--|:--|:--|:--|",
    ]
    for case, (k, name, kernel) in BM_CASES.items():
        if case in names:
            lines.append(
                f"| {k} · {name} | `{case}` | {kernel} | [RESULTS.md](BM/{case}/RESULTS.md) |"
            )
    lines += [
        "",
        "## Composite plate",
        "",
        "Rows: number of ply angles m. Columns: training size n.",
        "",
        "| m \\ n | " + " | ".join(str(n) for n in COMPOSITE_SIZES) + " |",
        "|:--|" + ":--:|" * len(COMPOSITE_SIZES),
    ]
    for m in COMPOSITE_DIMS:
        cells = [
            f"[m={m}, n={n}](COMPOSITE/Composite_{m}_{n}/RESULTS.md)"
            if f"Composite_{m}_{n}" in names
            else "–"
            for n in COMPOSITE_SIZES
        ]
        lines.append(f"| **{m}** | " + " | ".join(cells) + " |")
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def write_all(selection: list[str] | None = None) -> tuple[int, int]:
    cases = case_dirs(selection)
    n_img = sum(write_case(c) for c in cases)
    write_index(case_dirs())
    return len(cases), n_img


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cases", help="comma-separated case folders (default: all)")
    a = ap.parse_args()
    n_cases, n_img = write_all(a.cases.split(",") if a.cases else None)
    print(f"RESULTS.md written for {n_cases} cases ({n_img} images) and at the root")


if __name__ == "__main__":
    main()
