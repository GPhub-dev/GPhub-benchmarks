# Generic composite-plate scripts

One script per library, valid for every input dimension `m` (read from `train_x`). `run_campaign.py --sync`
copies this folder into every `COMPOSITE/Composite_<m>_<n>/scripts`.

Libraries: `DACE`, `egobox-gp`, `GPflow`, `GPML`, `GPstuff`, `GPyTorch`, `OpenTURNS`, `SMT`, `tinygp`, `UQLab`.

Harmonized specification (Section 4.5 of the paper):

| Setting | Value |
|---|---|
| Trend | constant, estimated |
| Kernel | Gaussian, anisotropic: R = exp(-1/2 sum_k (h_k / l_k)^2), process variance estimated |
| Initial length-scale | l0 = 0.25 sqrt(m) for every input of the [0, 1]-scaled design |
| Bounds | l in [1e-2, 1e2] in the libraries that expose bounds (DACE, egobox-gp, OpenTURNS, SMT, UQLab); GPflow, GPML, GPstuff, GPyTorch, and tinygp optimize unconstrained parameters |
| theta-libraries (DACE, SMT, egobox-gp) | R = exp(-sum_k theta_k h_k^2) on inputs standardized by the library: theta = var(x) / (2 l^2), var(x) = 1/12 |
| UQLab | length-scale on inputs standardized by the library: l / sd(x), sd(x) = 1/sqrt(12) |
| Nugget | 1e-8, fixed where the library exposes it (a noise variance, where it is a hyperparameter, is held at 1e-8) |
| Estimation | maximum likelihood, optimizer native to the library |
| Starts | one optimizer run from l0 (multi-start options are switched off); at most 1000 iterations where the library exposes a cap |

Notes

- `lib_OpenTURNS.py` disables the status check of the TNC optimizer, so that a model is returned when its line search stops early.
- `lib_GPstuff.m` uses `prior_logunif` so that the optimized energy is the negative log-likelihood (GPstuff optimizes the log-parameters).
- `lib_GPyTorch.py` leaves `max_cholesky_size` at its default (800): for n = 1000 the library uses conjugate gradients, as reported in the paper.
- The composite plate is run for m in {2, 4, 8, 16, 32, 48}.
