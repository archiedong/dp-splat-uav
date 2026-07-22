# Phase A engineering notes

Integration-audit items carried forward (correctness unaffected; tests handle them):

1. **Shared-prior precondition.** Cross-tile merge exactness requires all tiles to share
   one global NIW prior lam_0. `weighted_fit` currently defaults to tile-local priors;
   callers must install the shared global prior (see tests' `State._replace` pattern).
   The Phase-A pipeline runner must own this; consider a `prior=` parameter on the
   weighted-fit API.
2. **Which alpha at merge time.** `eppf_penalty` takes alpha as an argument; the pipeline
   must pass alpha_global (not alpha_t), per the sum_t alpha_t = alpha_global bookkeeping.
3. **Zero-count components.** `eppf_penalty` documents N > 0; the -inf limit fails safe,
   but the pipeline should prune zero-count components before matching.

4. **Memory ceiling at ~170M points.** smbu_full (169M pts) peaked at 37.6 GB RSS on the
   36 GB machine (completed via compressed memory). Larger scenes need chunked LAZ/LAS
   loading straight into per-tile buffers, or float32 staging before the float64 fit
   arrays. Not blocking for the paper's current dataset slate.
5. **Cross-tile merge acceptance is rare at 1e7-point tiles** (0 accepted on all three
   H3D train epochs and SMBU; 1 on the val split). Consistent with the conservative
   sign test at large N. Halo-width / threshold ablation (P2-Q7 item 1's ELBO-check
   arm) should quantify this before the paper claims merge behavior.
