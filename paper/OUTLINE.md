# Paper 2 OUTLINE — "DP-Splat for UAV/Aerial Mapping with Calibrated Uncertainty" (working title)

> **SUPERSEDED (2026-07-20).** This outline is retained as a planning record only. The
> compiled paper (`paper/main.tex` + `paper/sections/*.tex`, built with
> `cd paper && tectonic main.tex`) is the sole source of truth for claims, numbers, and
> registrations. Known staleness (audit §E): the claims table below predates the
> 2026-07-17 post-referee normalization correction (C1's "+1.67/+2.17/+2.20" seam gaps
> are superseded — stitched ≈ crop parity; see PHASE_B_FINDINGS.md and BRIEF.md §7),
> §V registrations and the figure inventory no longer match the JSONs, and Nov18/Mar19
> point counts are swapped. Do not re-register here; number provenance now runs
> paper text → `experiments/out/*.json` directly. Not deleted, per record-keeping rules.

Governing docs: `../paper2-uav/BRIEF.md` (adopted v1.0), `../paper2-uav/PHASE_B_FINDINGS.md`
(all numbers), `../paper2-uav/LIT_REVIEW.md` (references + novelty paragraph of record).
Venue: IEEE TGRS, `\documentclass[journal]{IEEEtran}`, two-column, 10 pages target incl.
references. Compile: `cd paper && tectonic main.tex`. Single author: Aqi Dong, Embry-Riddle
Aeronautical University, donga2@erau.edu. Method source citation: arXiv:2607.10912 (Paper 1).
Zero content reuse from Paper 1.

**Number provenance rule (binding):** every number below is traceable to
`PHASE_B_FINDINGS.md` or a file in `experiments/out/`. Section writers may not introduce
any number not in this outline without adding it here first with a source file.

**K̂ framing (binding, BRIEF §0):** K̂ = automatic per-tile capacity adaptation to data
volume. NEVER a scene-content complexity map. This overrides the last sentence of the
LIT_REVIEW novelty paragraph ("licensed spatial complexity map") — that clause is struck;
the rest of the paragraph is used verbatim-adapted in §I/§II.

---

## 0. CLAIMS–EVIDENCE TABLE (the paper's load-bearing claims; nothing else gets claimed)

| # | Claim (as the paper states it) | Exact evidence | Source |
|---|---|---|---|
| C1 | Independent per-tile DP-Splat fits + exact conjugate matched merging + partition-of-unity stitching yield a single normalized predictive density that beats hard seam heuristics (crop; Voronoi) on true held-out log-predictive/point, on every H3D epoch | Held-out complement protocol, production config (α=100, T=256): stitched−crop gap = +1.67 (Mar18), +2.17 (Mar19), +2.20 (Nov18) nats/pt; at α=1: +1.62/+2.12/+2.14. Mechanism: hard rules delete genuine straddling components (3 dropped on val scene, one carrying 261k points) | PHASE_B_FINDINGS §"Cross-epoch replication" + §"Honest-holdout"; `out/mar18_rich_eval.json` (stitched −4.150 vs crop −5.818), `mar19_rich_eval.json`, `nov18_rich_eval.json`, `*_ho_eval.json` |
| C2 | The stitched predictive is well calibrated at production capacity, including near seams | 68% coverage 0.696 all / 0.669 seam (Mar18), 0.685/0.645 (Mar19), 0.685/0.638 (Nov18); 90%: 0.935/0.928, 0.930/0.908, 0.930/0.911; 95%: 0.976/0.975, 0.972/0.965, 0.973/0.968. The α=1 seam undercoverage at 68% (0.499/0.470/0.445) is a capacity artifact, resolved at production | PHASE_B_FINDINGS §"Cross-epoch replication" + capacity study item 2; `out/*_rich_eval.json`, `out/*_ho_eval.json` |
| C3 | Predictive NLPD is a provably ideal error ranker under correct specification and a genuinely better-than-random ranker on real LiDAR | On model-matched synthetic truth: model−ideal AUSE = 0.0000, Spearman vs true NLPD = 1.0; AUSE math verified to 1e-12 vs independent implementation. Held-out H3D: AUSE 0.171/0.219/0.243, ρ(u,\|e\|) = 0.573/0.534/0.508 (production, three epochs); α=1 range AUSE 0.16–0.27, AURG 0.43–0.63 | PHASE_B_FINDINGS §"Phase B-1"; `out/*_ho_sparsification.json`, `out/*_rich_sparsification.json` |
| C4 | The uncertainty signal decomposes honestly: a causally meaningful density component plus residual model signal; still better than random at every density level | ρ(u, density) = −0.56 to −0.70; density-tercile-stratified AUSE 0.26–0.48 (all terciles < random). Capacity-insensitive: AUSE 0.171 vs 0.163 at K̂ 45 → 841 — confound is intrinsic to LiDAR-vs-Gaussian misspecification, not resolution | PHASE_B_FINDINGS §"Phase B-1" + capacity study item 1; `out/mar18_*_sparsification.json` |
| C5 | Model-derived LoD95 change detection is null-calibrated on physically stable ground and detects labeled change above chance, with a full capacity dose-response | Null exceedance 4.1% vs nominal 5% on 17,592 stable impervious cells (median LoD 1.157 m on null cells); insensitive to a 2 cm registration term. Veg-excluded AUC 0.72 / recall 0.21 at 1 m cells; 0.78 / 0.35 at 2 m (null 4.7%); 0.68 at 0.5 m. Dose-response K̂ 45 → 3,443 (α=1000/T=1024): AUC 0.72→0.78 at 1 m, precision 0.08→0.21, null stays conservative (0.9%); LoD floor ≈1.0 m intrinsic | PHASE_B_FINDINGS §"Phase B-2"; `out/change_full_nov18_mar19.json` (null 0.04133, AUC 0.7168, n_null=17,592, median_lod 1.845 m valid cells) |
| C6 | Redundant re-flights are worthless at any dose; posterior uncertainty finds where the model is wrong; uncertainty × recoverable mass says where re-flying pays — out-planning a per-tile error oracle | Interleaved-holdback null: no arm (incl. oracle) moves held-out NLPD beyond ±0.03 at any dose. Block scenario: targeted +0.53 vs random +0.41 ± 0.13 (5 seeds) vs oracle +0.44; locality decomposition +1.44 (granted core) / +0.65 (halo) / exactly 0.00 (untouched tiles). Yield planner (12 tiles, 8 seeds): expected yield = tile mean NLPD × grantable pool mass → +0.985 held-out, above oracle (+0.457) and above all 8 random draws (best 0.846; range 0.003–0.846); raw-NLPD ranking is mass-blind (+0.099 overall despite +2.60 local) | PHASE_B_FINDINGS §"Phase B-3"; `out/reflight_rubble.json`, `reflight_rubble_50.json`, `reflight_rubble_block.json`, `reflight_rubble_block12.json`, `reflight_rubble_yield12.json` |
| C7 | The whole pipeline is laptop-scale on 10⁸-point scenes | H3D epochs (59M/119M/129M pts): 274–612 s each; SMBU merged cloud 169M pts, 20 tiles: 863 s, peak RSS 37.6 GB; Mar18 production fit 705 s for 53.5M pts; ELBO monotone on every fitted tile, all scenes | PHASE_B_FINDINGS §"What worked" + capacity study; `out/*_record.json`, `out/smbu_full_record.json` |
| C8 | K̂ adapts capacity automatically to per-tile data volume, unlike fixed-K budgets — and (honestly) is NOT a scene-content complexity meter | STPLS3D matched-N grid (29 scenes × α∈{0.1,1,10} × 2 seeds, N=3×10⁶): at α=1, K̂∈[18,25] across empty-terrain→dense-urban; ρ(K̂, class-entropy)=0.13 (p=0.34), ρ(K̂, surface variation)=−0.04, ρ(K̂, density)=0.00. α=10 arm truncation-pinned at T=64 (all 58 fits). H3D: ρ(K̂, tile n_points)=0.97/0.76/0.76. Salvage probe: Ψ₀ over 4 orders of magnitude, K̂ unchanged to the integer; 261 controlled fits total | PHASE_B_FINDINGS §"deconfound" + salvage note; `out/stpls3d_full_grid.json`, `out/stpls3d_salvage_grid.json`, `out/*_record.json` |
| C9 | The cross-tile matcher is a live mechanism exactly when models are rich enough to have straddling structure | At production capacity: 42 accepted pairs → 37 m-way merge sets (K̂ 880→841) on Mar18; at α=1: 0–1 merges (0 on all train epochs and SMBU; 1 on val) — consistent with the conservative sign test at large N | PHASE_B_FINDINGS capacity item 4 + "Remaining protocol notes"; `out/mar18_rich_record.json`, `out/full_val_record.json` |
| C10 | Absolute predictive quality improves +2.5 nats at production capacity vs α=1 | Stitched −4.15 (production) vs −6.62 (α=1), Mar18 held-out | PHASE_B_FINDINGS capacity item 3; `out/mar18_rich_eval.json` vs `out/mar18_ho_eval.json` |

**Global banned-claims list (BRIEF §4 + LIT_REVIEW; applies to every section):**
novel rasterization; SOTA novel-view synthesis; real-time; any PSNR/LPIPS number or
benefit; metric depth without anchor validation; "first Bayesian/NIW splatting"; "first
DP complexity control for splats"; "first uncertainty-driven active splatting" (GAVIS);
"first drone deployment" (GauSS-MI flew one); "no prior work handles overlap" (say:
"duplication is avoided by deletion, never resolved by fusion"); "moment-matched merging"
(say: prior-subtracted natural-parameter merging, exact); K̂ as scene-content/semantic
complexity map (say: data-volume-adaptive capacity control); Mega-NeRF inference-time
blending (misattribution); Kerbl LOD interpolation as a seam mechanism; blanket "VBGS
lineage ignores uncertainty"; "first conjugate/CAVI NIW mixture on colored points" (VBGS
is); "first streaming conjugate splatting"; precision at LoD95 as a false-alarm rate (the
null test carries that role); any theoretical seam-coverage guarantee (empirical checks
only); image-space calibration ownership (Jia et al. own it; ours is point/density-space).

---

## I. Introduction (~1 page)

**Narrative:** aerial mapping products increasingly feed decisions (inspection, change
monitoring, re-flight planning) that need trustworthy, spatially resolved uncertainty —
not post-hoc heuristics. Large aerial scenes force partitioned fitting; the one-sentence
story of record (BRIEF §0, revised 2026-07-16): exact conjugate merging turns independent
tile posteriors into one coherent, calibrated predictive density — beating the field's
hard seam heuristics — and the posterior variance tells the operator where the
reconstruction cannot be trusted, and where to fly next.

**Claims made here (forward references only, each anchored in §V):**
- C1 (headline method result), C2 (calibration), C3/C6 (utility: ranking + planning), C7 (laptop scale).
- Novelty positioning: use LIT_REVIEW integration memo §2 paragraph with the K̂ clause
  struck (see binding note above) and Track 2's honest seam sentence.

**Contributions list (4 items, each mapping to a claims row):**
1. Tiled/partitioned DP-Splat inference with halo down-weighting, α_t bookkeeping
   (Σα_t = α_global), exact matched merge, and mass-conserving partition-of-unity
   stitching → C1, C9.
2. First exact-conjugate-generative treatment of tiled large-scene splatting evaluated
   as a normalized predictive density, with held-out seam-aware evaluation protocol
   beating crop/Voronoi baselines → C1, C2.
3. Calibrated point/density-space uncertainty shown useful on three remote-sensing tasks:
   sparsification (C3, C4), null-calibrated change detection with dose-response (C5),
   and yield-based re-flight planning that out-plans an error oracle (C6).
4. Laptop-scale practicality on 10⁸-point scenes (C7) with automatic per-tile capacity
   adaptation (C8) — honestly deconfounded.

**Figures/tables:** none (or a teaser figure only if page budget allows — candidate:
`figures/mar18_train_khat_map.pdf` recaptioned as capacity map; caption must say
"capacity tracks per-tile point count (ρ=0.97)", NOT complexity).

**Banned here:** all global bans; especially do not oversell K̂ or claim firsts covered
by Jia/VBGS/GAVIS.

## II. Related Work (~1.25 pages)

Four sub-blocks, each with its pre-approved sentence and citation set (LIT_REVIEW):

**II-A Tiled large-scale splatting/NeRF.** Use verbatim-adapted the honest seam sentence
(LIT_REVIEW Track 2 §"The sentence Paper 2 can honestly write"): crop (VastGaussian
CVPR 2024 Sec. 4.3; CityGaussian ECCV 2024 Sec. 3.2 + merge script; Kerbl et al. TOG 2024
Sec. 6.3 Voronoi cull), train-time ADMM consensus with unweighted 1/K averaging (DOGS
NeurIPS 2024 Eq. 8b — the foil for our merge), image-space IDW (Block-NeRF CVPR 2022
Sec. 4.3; appearance-matching as structural warning). Mega-NeRF cited for partitioning
only — NO inference-blending attribution. Phrase of record: "duplication is avoided by
deletion, never resolved by fusion."

**II-B Bayesian splatting.** Jia et al. arXiv:2607.05522: overlay-vs-generative
paragraph — their NIW summaries are optimization-driven surrogates (their Sec. 4
concession), no assignments/E-step/joint ELBO, DP sticks as pruning only; they own
image-space calibration (cite as precedent; ours is point/density-space). VBGS
(arXiv:2410.03592, cite as preprint, venue unconfirmed): conjugate CAVI but fixed
finite K, no predictive use of posterior; VBGS-SLAM is UC Riverside (not VERSES),
propagates pose uncertainty — distinguish, don't blanket-dismiss. Paper 1
(arXiv:2607.10912) as method source.

**II-C Aerial 3DGS cluster.** Ready-made gap statement (LIT_REVIEW Track 3 §4):
Yan et al. 2025 (RS 17(23):3868 — cite as 2025), Ortho-3DGS (JSTARS 18, 2025), ULSR-GS
(arXiv-only, flag), ARSGaussian, Horizon-GS, TreeDGS, Atik 2025, Fadilah 2026 (their
"3DGS-MCMC" is a densification sampler — word carefully). GU-GS (TGRS 2026):
uncertainty = loss reweighting; abstract-level claims only (full text unverified —
one-sentence cite max). PRIMU (arXiv:2508.02443): post-hoc regressor, sets AUSE
vocabulary, reports no calibration → coverage is our differentiator.

**II-D Uncertainty evaluation + distributed BNP.** AUSE (Ilg et al. ECCV 2018), AURG
(Poggi et al. CVPR 2020), M3C2 (Lague et al. 2013), M3C2-EP = Winiwarter, Anders & Höfle
2021 (NOT Zahs 2022 — that is CD-PB-M3C2), Huang & Qin (ISPRS Archives 2025). Merge
machinery citation trail ("we adapt, not invent"): Campbell et al. 2015 (EPPF matching),
Hughes & Sudderth 2013 (merge ratio), SPAHM 2019 (Hungarian matching), Scott et al. CMC
2016, Williamson et al. 2013 Thm. 1 (α bookkeeping), Dunson & Park 2008 KSBP
(compact-kernel limit = independent tile DPs; cite as model anchor for tiling, NOT for a
complexity-map claim). Active cluster cited without firsts: GAVIS, FisherRF, ActiveSplat,
GauSS-MI (venue "RSS 2025" per GitHub — flag), NVF (cite via GAVIS ref [104]).

**Figures/tables:** none.

**Banned here:** all firsts listed in the global bans; unverified-flag items (PL4U,
GU-GS full text, NVF, GauSS-MI venue, Reich & Fuentes, Quintana, FedMA, Singh & Jaggi)
at more than one-sentence level; Block-NeRF visibility threshold number (not in their
paper); Gelfand et al. 2005 as an anchor (it is affirmatively the wrong one — may be
cited as a contrast).

## III. Method (~2 pages)

**III-A Model recap (≤1 page hard cap, freshly written, cites Paper 1).** DP mixture of
NIW-conjugate Gaussians on colored points (spatial ⊗ color modalities), truncated
stick-breaking, CAVI/SVI with monotone ELBO, Student-t predictive. No equations
reproduced beyond what tiling needs; everything else "see arXiv:2607.10912".

**III-B Tiled weighted fitting.** Grid partition with overlap halos; halo points enter
suff-stats down-weighted by 1/#owning-tiles; per-tile α_t ∝ tile count with
Σ_t α_t = α_global (Williamson et al. 2013 Thm. 1 licenses global-K̂ bookkeeping);
embarrassingly parallel per-tile fits. Method description must match
`experiments/run_tiled_fit.py` (production defaults: α=100, T=256, halo from records).

**III-C Exact matched merge.** For adjacent tiles: score halo-adjacent component pairs
by the closed-form NIW log-marginal ratio ΔL = A(λ_A+λ_B−λ_0) − A(λ_A) − A(λ_B) + A(λ_0)
per modality; Hungarian one-to-one assignment; Campbell-et-al. EPPF penalty as threshold;
merge matched sets by prior-subtracted natural-parameter addition λ_merged = Σλ_i − (m−1)λ_0
(m-way allowed; exact conjugate arithmetic, NOT moment matching); never add Beta sticks
across tiles — rebuild weights from pooled counts (a_k = 1+N_k, b_k = α + Σ_{j>k}N_j).
Citation trail sentence "we adapt": Campbell+ 2015, Hughes & Sudderth 2013, SPAHM 2019,
Scott+ CMC, Williamson+ 2013. Note the conservatism-by-design (default to not merging on
thin halo evidence) and that acceptance activates at production capacity (C9 numbers in §V-A).

**III-D Partition-of-unity stitching.** p(x) = Σ_t g_t(s) p_t(x), g_t ≡ 1 on cores,
smooth blend in halos; mass conservation by construction (MC-checked ∫=1 on overlaps —
Phase A acceptance). Ships regardless of merge decisions.

**III-E K̂ as capacity adaptation.** One subsection, honestly framed: per-tile K̂ adapts
to data volume automatically (vs fixed-K budgets that waste capacity on empty tiles and
starve dense ones — VBGS foil); explicitly state it is NOT a scene-content complexity
map, forward-referencing the §V-B deconfound and §VI. Per-tile K̂ reported pre-merge;
global K̂ post-merge (P2-Q7 item 3, confirmed).

**Figures/tables:** one method schematic (tiles/halos/merge/stitch — TO BE MADE, does
not exist in figures/ yet; the only new figure this outline authorizes).

**Banned here:** "moment matching"; any global-ELBO-monotonicity claim for the merged
model (halo-local objective caveat, LIT_REVIEW Track 1 Rank 4); silent equation changes
(any new equation needs QUESTIONS.md + NumPy oracle test — all five Phase A equations
already have them, suite 99 green).

## IV. Experimental Setup (~1 page)

**IV-A Datasets + licenses.**
- H3D Hessigheim (registration-gated; three LiDAR epochs used: Mar18 59M / Nov18 119M /
  Mar19 129M points — exact counts from `out/*_record.json`; labels used for the change
  proxy). License: CC BY-NC-SA variant — we release code + pipeline scripts only, never
  derived point clouds (BRIEF ground rule 1).
- Mill 19 Rubble (Mega-NeRF, un-gated): 1,678 images at 4× downscale, shipped PixSfM
  poses fixed; CPU SIFT → sequential+spatial matching → COLMAP point_triangulator
  (OPENCV model, no pose BA) → 1,750,606-point sparse anchor; robust extent 274.7 × 317.9 m
  footprint; camera grid 206.5 × 248.3 m, altitude spread 5.1 m (`out/mill19_anchor_report.json`).
  Label as *pipeline product*, not dataset ground truth (ground rule 3).
- STPLS3D synthetic (un-gated): 29 scenes, matched N = 3×10⁶ per fit, for the capacity
  deconfound grid (`out/stpls3d_full_grid.json`, `stpls3d_salvage_grid.json`).
- SMBU / GauU-Scene V2 merged photogrammetric cloud (169M pts) — scale demonstration
  only (`out/smbu_full_record.json`).

**IV-B Production configuration.** α=100, T=256 per tile; grid tiling with halos; SVI
path; all fits CPU (Apple laptop, 36 GB); config recorded per run in
`out/*_record.json` (writers: pull halo width, target tile points, iteration counts
from the records, not from memory).

**IV-C Protocols.**
- Honest-holdout complement: refit on recorded 90% subsample (seed 0), evaluate 200k
  points drawn from the exact complement (seed 11) — true held-out
  (`out/*_ho_eval.json`, `*_rich_eval.json` holdout_note field).
- Seam definition + seam-restricted scoring (seam fraction ~8.5%, e.g. n_seam=16,941 of
  200k on Mar18 — `mar18_rich_eval.json`).
- Baselines: VastGaussian-style mean-in-tile crop; Kerbl-style Voronoi ownership; note
  crop ≡ Voronoi exactly on equal-grid layouts (proven; would differ on irregular
  partitions) — so the table reports one column for both where identical.
- Change protocol: model-derived LoD95 with Σ_μ = Ψ_N/(κ_N(ν_N−D−1)) (posterior MEAN
  covariance, not predictive — M3C2-EP double-counting pitfall avoided by construction;
  MC-verified); LoD = 1.96√(var_a+var_b+σ_reg²); null set = physically stable impervious
  cells; class-transition proxy with veg-excluded scoring (`out/change_full_nov18_mar19.json`).
- Re-flight protocol: leakage-proofed (structural proof: no eval point enters any refit,
  any arm); interleaved vs block holdback; arms = uncertainty-targeted, random (5–8
  seeds), per-tile error oracle; yield score = tile mean NLPD × grantable pool mass
  (`experiments/reflight.py`, `out/reflight_*.json`).

**IV-D Metrics.** Held-out log-predictive/pt (all + seam); central-interval coverage at
68/90/95%; AUSE/AURG (verified vs independent implementation to 1e-12) + Spearman
ρ(u,|e|) + density-stratified variants; change AUC/recall/precision + null exceedance;
re-flight held-out NLPD deltas with core/halo/untouched decomposition.

**Figures/tables:** Table: dataset summary (scene, points, tiles, fit seconds, peak RSS
for SMBU). All numbers from `out/*_record.json` and PHASE_B_FINDINGS §"What worked".

**Banned here:** calling any computed cloud "ground truth"; M3C2 as a scoring metric
(radius sensitivity — we use it nowhere; don't imply we do); claiming the class-transition
proxy measures false alarms.

## V. Results (~3 pages)

### V-A Seam handling: matched merge + stitching vs crop/Voronoi (held-out)

Claims: C1, C9, C10, C7.
- MAIN TABLE (the paper's Table 2): three epochs × {stitched, crop=Voronoi} × {all, seam}
  held-out log-predictive at production config, plus K̂ global and gap columns:
  Mar18 stitched −4.150 / crop −5.818 (gap +1.67), K̂ 841; Mar19 gap +2.17, K̂ 1050;
  Nov18 gap +2.20, K̂ 1108. Sources: `out/mar18_rich_eval.json`,
  `mar19_rich_eval.json`, `nov18_rich_eval.json`. Include α=1 held-out row or footnote
  (+1.62/+2.12/+2.14, `out/*_ho_eval.json`) to show robustness across capacity.
- Mechanism sentence: hard rules delete genuine straddling components (3 dropped on val,
  incl. one carrying 261k points) — PHASE_B_FINDINGS §"What worked" item 2.
- Merge activation: 42 accepted pairs → 37 m-way merge sets, K̂ 880→841 (production
  Mar18); 0–1 accepted at α=1 (0 on all train epochs and SMBU; 1 on val) — presented as
  the sign test being conservative at large N, a feature (C9).
- Runtime/scale paragraph (C7): 274–612 s per H3D epoch; SMBU 169M pts / 20 tiles /
  863 s / 37.6 GB peak RSS; production Mar18 705 s; zero ELBO dips on any fitted tile.
- Figures: none required (table-led); optional `figures/smbu_full_khat_map.pdf` as the
  scale demo (caption = capacity framing).

Banned: "no prior work deduplicates"; PSNR anything; claiming merge acceptance rate
should be high (conservatism is by design).

### V-B Calibration incl. capacity study

Claims: C2, C10.
- Coverage table (or merged into Table 2): 68/90/95% all/seam, three epochs, production
  config (numbers in C2 row). Source: `out/*_rich_eval.json` coverage blocks
  (axis_mean values; per-axis x/y/z available for an appendix-level remark — z runs
  conservative, x/y slightly under at 68%).
- Capacity study narrative (Mar18, K̂ 45 → 841): the α=1 seam 68% undercoverage
  (0.659/0.499 → 0.696/0.669) was a capacity artifact — coarse components straddling
  seams; near-nominal everywhere at production. +2.5 nats absolute (−4.15 vs −6.62).
  Sources: `out/mar18_ho_eval.json` vs `out/mar18_rich_eval.json`.
- Honest note: 90/95% mildly conservative everywhere (0.925–0.976); 68% slightly under
  overall — report, don't hide.
- Figures: none existing for coverage; a small coverage-vs-nominal table suffices
  (page budget). Do NOT promise a reliability diagram unless one is generated and added
  to figures/ first.

Banned: theoretical seam-coverage guarantee (empirical only — P2-Q7 item 6);
"calibrated" without stating level-specific numbers.

### V-C Sparsification with density decomposition + on-spec ideality

Claims: C3, C4.
- Verification lead: AUSE machinery verified to 1e-12; on model-matched synthetic truth
  NLPD ranker attains the irreducible floor EXACTLY (model−ideal = 0.0000, Spearman 1.0).
  "The machinery is right; what follows is about the model on misspecified real LiDAR."
- Main numbers (production, held-out, three epochs): AUSE 0.171/0.219/0.243,
  ρ(u,|e|) 0.573/0.534/0.508; α=1 ranges AUSE 0.16–0.27, AURG 0.43–0.63,
  ρ 0.51–0.63. Sources: `out/*_rich_sparsification.json`, `out/*_ho_sparsification.json`.
- Density decomposition (the honesty centerpiece): ρ(u,density) = −0.56 to −0.70;
  tercile-stratified AUSE 0.26–0.48, better than random at every density level.
  Capacity-insensitivity (AUSE 0.171 vs 0.163; terciles slightly worse) → confound is
  intrinsic to LiDAR-vs-Gaussian misspecification. Report the dead variant honestly:
  per-component sqrt-variance is uninformative (piecewise-constant).
- Supportable claim sentence (verbatim from PHASE_B_FINDINGS): predictive NLPD is
  provably ideal under correct specification and a useful real-data error ranker whose
  signal decomposes into a causally meaningful density component plus residual model
  signal.
- Figures (exist): `figures/mar18_rich_sparsification.pdf`,
  `mar19_rich_sparsification.pdf`, `nov18_rich_sparsification.pdf` (production; prefer
  these) — one epoch in main text, others referenced or in a combined panel; `*_ho_*`
  variants available for the α=1 comparison if space allows.

Banned: implying the ranker beats PRIMU/GU-GS numerically (no common benchmark);
hiding the density confound; claiming image-space calibration.

### V-D Change detection with null test + dose-response

Claim: C5.
- Setup recap: Nov18 ↔ Mar19, production fits, model-derived LoD95, 1 m cells, 20 s
  runtime; grid 313×783 cells, 113,020 valid, 17,592 null (stable impervious),
  43,221 eligible veg-excluded, 734 positives (`out/change_full_nov18_mar19.json`
  n_cells block).
- Null calibration: 4.1% exceedance vs nominal 5%; insensitive to 2 cm registration
  term; synthetic 30 cm slab: recall 1.000, null flag rate 0.000, |d| error 0.06%.
- Detection: AUC 0.717 / recall 0.21 at 1 m; 0.78 / 0.35 at 2 m (null 4.7%); 0.68 at
  0.5 m. Median |d| 0.279 m, p95 2.47 m, median LoD 1.845 m (valid cells).
- Dose-response (capacity ablation, closed): K̂ 45 → 3,443 (α=1000/T=1024): AUC
  0.72→0.78 at 1 m, precision 0.08→0.21, null 0.9% (still conservative); LoD floor
  ≈1.0 m intrinsic to Gaussian-mixture representation of heterogeneous urban vertical
  structure — not component footprint, cell size, or registration.
- Mandatory proxy caveat (verbatim adaptation): class transitions undercount true
  vertical change (same-class construction) and include vertical-less changes
  (vehicles) — precision at LoD95 is NOT interpretable as false-alarm rate; the null
  test carries that role.
- Figures (exist): `figures/change_full_nov18_mar19.pdf` (main);
  `change_nov18_mar19.pdf` available as the earlier/smaller run — use full version only.

Banned: precision-as-false-alarm-rate; M3C2-EP double-counting (we avoid it by
construction — say so, cite Winiwarter 2021 correctly); overclaiming sub-meter change
sensitivity (floor is ≈1.0 m).

### V-E Re-flight: three-act structure + yield acquisition

Claim: C6.
- Anchor: Mill 19 Rubble sparse COLMAP (1.75M colored points, 1,678 fixed-pose images,
  CPU-only, 275×318 m site); leakage-proofed protocol (structural proof stated).
- Act 1 — redundant re-flights are worthless: interleaved holdback grants redundant
  views of covered ground; no arm, including a per-tile oracle, moves held-out NLPD
  beyond ±0.03 at any dose (two null runs kept as controls). Operational reading:
  re-fly NEW ground. Sources: `out/reflight_rubble.json`, `reflight_rubble_50.json`.
- Act 2 — uncertainty finds where the model is wrong: block scenario (contiguous
  unfinished 20% strip): targeted +0.53 vs random +0.41 ± 0.13 (5 seeds) vs oracle
  +0.44; locality decomposition +1.44 core / +0.65 halo / exactly 0.00 untouched
  (also falsifies weight-dilution concern; α_t effects 6th-decimal). Sources:
  `out/reflight_rubble_block.json`, `reflight_rubble_block12.json`.
- Act 3 — uncertainty × recoverable mass says where re-flying pays: raw-NLPD ranking is
  mass-blind (+2.60 local, +0.099 overall, below random mean +0.401 ± 0.33, itself
  wildly variable 0.003–0.846); expected yield = tile mean NLPD × grantable pool mass
  → +0.985 held-out, above the per-tile oracle (+0.457) and above every one of 8
  random draws (best 0.846). Closing line: the model's own posterior out-plans an
  error oracle. Source: `out/reflight_rubble_yield12.json`.
- Figures (exist): `figures/reflight_rubble_yield12.pdf` (main result);
  `reflight_rubble_block.pdf` or `reflight_rubble_block12.pdf` (Act 2);
  `reflight_rubble.pdf` / `reflight_rubble_50.pdf` (Act 1 nulls — one suffices or
  fold into text).

Banned: "first uncertainty-driven active splatting" / any NBV-first; claiming flight
execution (this is a planning experiment on held-back imagery); PSNR framing; claiming
the oracle is globally optimal (it's a per-tile error oracle, out-selected via eval-mass
weighting — say exactly that).

## VI. Discussion / Limitations (~0.75 page)

Each limitation stated with its evidence line — this section is where reviewer trust is won:
1. **LoD floor is intrinsic** (≈1.0 m): survives capacity ×76 (K̂ 45→3,443), cell-size
   sweep, and registration term — a representation limit of Gaussian mixtures on
   heterogeneous urban vertical structure (C5 evidence).
2. **Density confound in uncertainty ranking**: ρ(u,density) −0.56 to −0.70; reported
   with tercile stratification; capacity-insensitive → intrinsic to point-density
   heteroscedasticity vs Gaussian likelihood (C4 evidence).
3. **K̂ deconfound reported honestly** (C8): the STPLS3D matched-N grid + salvage sweep
   (261 controlled fits, Ψ₀ over 4 orders, K̂ unchanged to the integer) establish that
   realized complexity is set by component-death dynamics and data volume under
   truncated-SB CAVI — capacity adaptation is the true, useful property; content mapping
   is explicitly disclaimed. Sources: `out/stpls3d_full_grid.json`,
   `stpls3d_salvage_grid.json`.
4. **NC data licensing**: H3D CC BY-NC-SA → code + scripts released, no derived clouds.
5. **Sparse-anchor scope**: Mill 19 results are on a sparse COLMAP anchor (1.75M pts) —
   a planning-scale demonstration, not a dense photogrammetric product; metric-honesty
   rule (BRIEF ground rule 2) — no absolute-scale claims beyond the COLMAP-sparse anchor.
6. Merge acceptance rarity at 10⁷-point tiles (α=1) and the halo/threshold ablation left
   to future work (P2-Q7 item 1 ELBO arm); HDP sharing out of scope (future work).
7. Seam-coverage: empirical checks only; no theoretical guarantee claimed.

**Figures/tables:** `figures/stpls3d_khat_vs_complexity.pdf` (exists — the deconfound
figure; caption states the null result: ρ(K̂, entropy)=0.13, p=0.34) and optionally one
K̂ map (`figures/mar18_train_khat_map.pdf`) paired with the ρ(K̂, n_points)=0.97 line so
the "capacity map" reading is visually honest.

Banned: spinning the deconfound as a positive complexity result; "future work will fix"
hand-waving on the LoD floor (it is intrinsic — say so).

## VII. Conclusion (~0.25 page)

Restate the one-sentence story (BRIEF §0). Three takeaways: (1) exact conjugate merging +
normalized stitching beats deletion-based seam handling by +1.6–2.2 nats held-out,
calibrated including seams at production capacity; (2) the posterior is operationally
useful three ways — error ranking, null-calibrated change detection, and yield-based
re-flight planning that out-plans an error oracle; (3) all of it runs on a laptop at
10⁸ points. Future work: halo/threshold ablations, HDP cross-tile sharing, dense
photogrammetric and depth-lifted sources (Phase D scope). No new claims here.

---

## Figure/table inventory (existing assets → planned slots)

| Slot | Asset (exists in figures/ unless noted) | Section |
|---|---|---|
| Fig. 1 | Method schematic — TO BE MADE (only authorized new figure) | III |
| Table 1 | Dataset/scale summary (from `out/*_record.json`) | IV |
| Table 2 | Held-out seam table + coverage (from `out/*_rich_eval.json`) | V-A/V-B |
| Fig. 2 | `mar18_rich_sparsification.pdf` (+ siblings as panel/appendix) | V-C |
| Fig. 3 | `change_full_nov18_mar19.pdf` | V-D |
| Fig. 4 | `reflight_rubble_yield12.pdf` (+ `reflight_rubble_block12.pdf` if space) | V-E |
| Fig. 5 | `stpls3d_khat_vs_complexity.pdf` | VI |
| Optional | `mar18_train_khat_map.pdf` / `smbu_full_khat_map.pdf` (capacity captions only) | I/V-A/VI |

## Writer checklist (every section)

1. Every number matches this outline; outline traces to PHASE_B_FINDINGS.md or
   `experiments/out/*.json`. No invented numbers, no rounding that changes a claim.
2. Check the global banned-claims list before submitting a section.
3. Flagged-unverified references: one-sentence cites max, as flagged in LIT_REVIEW.
4. Pipeline products labeled as such in every caption (BRIEF ground rule 3).
5. Voice: Paper 1 register (precise, honest, claims scaled to evidence); zero Paper 1
   content reuse; no AI/process references; single author Aqi Dong, ERAU.
6. K̂ language audit: "capacity", "data volume", "adaptation" — never "complexity map",
   "semantic", "content".

---

## ADDENDUM — §V number registrations (added 2026-07-17 by the results-section writer)

Per the number-provenance rule, the following numbers used in `sections/results.tex`
(beyond those already listed above) are registered here with sources:

- **Held-out log-predictive absolutes & seam splits, production** (`out/mar19_rich_eval.json`,
  `nov18_rich_eval.json`, `mar18_rich_eval.json`): Mar19 stitched −5.70 / crop −7.87;
  Nov18 −5.13 / −7.33. Seam subset: Mar18 −4.45/−5.93 (+1.48), Mar19 −6.10/−8.32 (+2.22),
  Nov18 −5.70/−7.88 (+2.19). Crop deletes 20/24/18 components (k_global 841/1050/1108 vs
  k_crop 821/1026/1090). n_holdout=200,000/epoch; seam fractions 8.5/8.2/8.6%.
- **α=1 reference arm absolutes & seam splits** (`out/*_ho_eval.json`): Mar18 −6.62/−8.24
  (seam −6.66/−7.91, +1.25); Mar19 −7.83/−9.95 (seam −8.50/−10.47, +1.96); Nov18
  −7.27/−9.41 (seam −7.86/−9.73, +1.87). K̂ of the 90%-subsample refits: 42/58/54.
  α=1 coverage 90% seam 0.843/0.825/0.835; 95% seam 0.959/0.953/0.949.
- **Production merge activation, Mar19/Nov18** (`out/mar19_rich_record.json`,
  `nov18_rich_record.json`): 29 pairs→29 sets (K̂ 1079→1050); 31→31 (1139→1108).
- **Production fit runtimes Mar19/Nov18** (`out/*_rich_record.json`): 1,344 s (107M-pt
  subsample), 1,878 s (116M-pt subsample). Reference-arm full-epoch fits 274/612/597 s
  for 59.4/119.0/128.9M pts (`out/*_train_record.json`). Held-out evaluation 24–94 s
  (`out/*_rich_eval.json` seconds_total).
- **Sparsification per-epoch scalars** (`out/*_rich_sparsification.json`,
  `out/*_ho_sparsification.json`): production AURG 0.576/0.510/0.482; ρ(u,density)
  −0.723/−0.639/−0.620. α=1: AUSE 0.163/0.267/0.247, AURG 0.612/0.466/0.479,
  ρ(u,|e|) 0.634/0.507/0.515, ρ(u,density) −0.702/−0.561/−0.570. Marginal-variance
  variant (sqrt_mean_marginal_var): AURG −0.04 to +0.15 across the six runs.
- **Change detection extras** (`out/change_full_nov18_mar19.json`): median |d| on null
  cells 0.174 m; flag rate on valid cells 3.7%; AUC under σ_reg=2 cm 0.7169 (vs 0.7168);
  runtime 20.6 s. **High-capacity change run** (`out/change_nov18_mar19.json`, built on
  `mar19_xrich_record.json`/`nov18_xrich_record.json`, α=1000/T=1024, K̂ 3,443/3,525):
  null exceedance 0.88%, AUC 0.779, precision 0.210, recall 0.174, median null-cell LoD
  1.026 m, median valid-cell LoD 1.568 m, median |d| 0.195 m.
- **Re-flight extras** (`out/reflight_rubble.json`, `_50.json`, `_block.json`,
  `_block12.json`, `_yield12.json`): interleaved 20%-holdback deltas: targeted +0.011
  (granted-tile +0.051), oracle −0.003 (granted-tile −0.018), random −0.005±0.012;
  50%-holdback: targeted = oracle +0.032 (identical selection; granted-tile +0.066),
  random +0.026±0.004. Block-6: baseline NLPD 6.258; oracle granted-tile
  delta +1.68; random granted-tile mean +1.23. Block-12: n_eval 63,005, baseline NLPD
  5.799; oracle granted-tile delta +1.76; random granted-tile mean +0.96; yield arm
  granted-tile delta +1.23, completeness +0.086; raw-NLPD arm grants only 8,931 pool
  points vs 157,739 (yield) and 54,503 (oracle).
