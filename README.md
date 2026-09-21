# GPhub-kit-benchmarks

Scripts, data, and stored results of the benchmarks of the paper *"GPhub: A comprehensive guide to Gaussian Process
libraries"* (Section 4), run with [GPhub-kit](https://github.com/GPhub-dev/GPhub-kit).

The benchmarks showcase the toolkit; they are neither a comprehensive comparison nor a ranking of the libraries.
**Browse the results: [RESULTS.md](RESULTS.md)** (one page per case, with the metrics and every plot).

They cover 10 libraries: `DACE`, `egobox-gp`, `GPflow`, `GPML`, `GPstuff`, `GPyTorch`, `OpenTURNS`, `SMT`, `tinygp`,
and `UQLab`. Contributions that add a library, or improve a script, are welcome.

## Layout

| Path | Content |
|---|---|
| `BM/BM_0{1,2,4,5,6,7}/` | Analytical test functions 1 to 6 of the paper (damped cosine, Branin-Hoo, Dette & Pepelyshev, OTL circuit, borehole, Morris) |
| `COMPOSITE/Composite_<m>_<n>/` | Composite plate, m in {2, 4, 8, 16, 32, 48} ply angles and n in {100, 200, 500, 700, 1000} training points |
| `<case>/data/` | Training and test sets (inputs scaled to [0, 1]) |
| `<case>/scripts/lib_<library>.*` | One script per library (Python, MATLAB) |
| `<case>/results/storage/` | Stored predictions, predictive variances, times, and peak memory (Parquet) |
| `<case>/results/img/`, `report.log` | Plots and report of the toolkit (radar charts, cross-validation, density, residuals, prediction) |
| `<case>/RESULTS.md` | The metrics table and every plot of the case on one page |
| `_scripts/` | Generic composite-plate scripts and their harmonized specification (`_scripts/README.md`) |
| `run_campaign.py` | Runs the cases library by library and logs every attempt in `campaign_log.jsonl` |
| `postprocess_all.py` | Regenerates the plots and reports of every case, then the `RESULTS.md` pages |
| `make_results_md.py` | Writes `<case>/RESULTS.md` (metrics table and every plot of the case) and the index [`RESULTS.md`](RESULTS.md) |

The scripts of the analytical functions use the native optimizer settings of each library; those of the composite
plate implement one common specification.

## Run

``` shell
PY=../GPhub-kit/.venv/bin/python                            # environment with gphubkit[all]
$PY run_campaign.py --sync                                  # copy _scripts/ into every COMPOSITE case
$PY run_campaign.py --group BM --timeout 1800               # 10 libraries x 6 analytical functions
$PY run_campaign.py --group COMPOSITE --timeout 1800        # 10 libraries x 30 composite cases
$PY run_campaign.py --cases BM_04 --libs GPflow,SMT         # any subset
```

- One process per case and library, strictly sequential, with a time limit of 30 min. Every attempt is logged in
  `campaign_log.jsonl` (status, library version, wall time); `results/logs/` holds one log per library.
- Never run two jobs at once and keep the machine idle: the BLAS threads of the Python stack spin-wait, and parallel
  runs inflate the timings by two orders of magnitude.
- Requirements beyond `gphubkit[all]`: MATLAB with DACE, GPML 4.2, GPstuff, and UQLab in `~/Documents/MATLAB`;
  `pip install "tensorflow-probability==0.24.0" "setuptools<81"` for GPflow.

## Plots and reports

``` shell
$PY postprocess_all.py                                      
$PY postprocess_all.py --cases BM_01,Composite_2_100
```

BM cases are written as PNG and PDF, COMPOSITE cases as PNG. Plotting is not timed and runs in parallel. The `RESULTS.md` pages are rewritten at the end; `$PY make_results_md.py` rewrites them alone.
