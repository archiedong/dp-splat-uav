# TGRS Submission Checklist — "Tiled Dirichlet-Process Gaussian Splatting for UAV Mapping with Calibrated Uncertainty"

Manuscript: `paper/main.pdf` (13 pages, IEEEtran two-column, single author Aqi Dong, ERAU).
Cover letter: `paper/submission/cover_letter.pdf`.

## Before uploading

- [ ] **Bio (PI-TODO):** replace the placeholder `IEEEbiographynophoto` block in
  `paper/main.tex` (degrees with institutions and years, current position,
  research interests). Recompile.
- [ ] **Received date:** fill the exact date in the first `\thanks{Manuscript
  received ...}` in `paper/main.tex`; add a funding line there if any.
- [x] **Code URL:** github.com/archiedong/dp-splat-uav live and cited; the data/availability paragraph now names the public
  release upon acceptance; if the `dp-splat-uav` repository is pushed before
  submission, replace the release clause with the URL
  (see `TODO-BEFORE-SUBMISSION` comment in `main.tex`).
- [ ] **ORCID:** ensure the author ORCID is current in the ScholarOne account —
  IEEE requires an ORCID for the submitting author.
- [ ] **Overlength decision (PI call, pending):** 13 pages vs. the 10-page
  no-fee limit; pages 11–13 billed at mandatory overlength page charges
  (~$690 non-member / ~$600 GRSS member at current rates — verify on the TGRS
  page-charge notice at submission). Layout analysis: cutting to 10 would
  remove ~3 pages of content (re-flight study, geometric-accuracy section, or
  K-hat deconfounding). Recommendation on file: keep 13 and pay.
- [ ] **GRSS membership (optional):** joining IEEE GRSS before submission
  reduces the overlength rate and is sometimes worth it on fee alone.
- [ ] **Final PDF check:** recompile `main.tex` with tectonic; confirm 0
  overfull boxes, no undefined references, page count unchanged; fonts
  embedded (verified via pdffonts on 2026-07-20: all 39 fonts embedded and
  subset, no Type 3).

## ScholarOne (mc.manuscriptcentral.com/tgrs)

1. Log in / create Author Center entry; verify ORCID is linked.
2. Start new submission: type **Regular Paper**.
3. Title, abstract (plain-text version of the LaTeX abstract), index terms
   (use the paper's IEEEkeywords list).
4. Upload files:
   - `main.pdf` as the main document (first submission may be a single
     review PDF; source files required at acceptance).
   - `cover_letter.pdf` as supplementary/cover letter.
5. Author step: single author Aqi Dong, ERAU affiliation, ERAU email.
6. **Suggested reviewers (verified against official pages, 2026-07-21;
   PI-approved selection criterion: constructive fit).** Primary four:
   - Rongjun Qin, Professor, Civil/Env. & Geodetic Eng. and ECE, The Ohio
     State University — qin.324@osu.edu (we adopt his uncertainty-evaluation
     protocol, ISPRS 2025).
   - Boris Jutzi, Prof. Dr.-Ing., IPF, Karlsruhe Institute of Technology —
     boris.jutzi@kit.edu (NeRF-ensembles/PRIMU, radiance-field point-cloud
     uncertainty).
   - Lukas Winiwarter, Univ.-Prof., Geodesy, University of Innsbruck —
     lukas.winiwarter@uibk.ac.at (M3C2-EP; our LoD extends his construction).
   - Dimitri Lague, CNRS Research Director, Geosciences Rennes / Univ.
     Rennes — dimitri.lague@univ-rennes.fr (M3C2 originator).
   Alternate (fifth slot only; mild COI optics as the H3D benchmark
   provider): Norbert Haala, apl. Prof., ifp, University of Stuttgart —
   norbert.haala@ifp.uni-stuttgart.de.
7. Declarations: not under consideration elsewhere; no conflicts; preprint of
   the *methods* paper exists (arXiv:2607.10912) and is cited — this
   manuscript itself has no preprint posted (post one only if PI decides;
   IEEE permits arXiv preprints).
8. Confirm the review-ready PDF renders correctly in ScholarOne's proof;
   approve and submit.

## IEEE PDF eXpress (required at acceptance; optional now)

- Site: pdf-express.ieee.org (conference/journal ID for TGRS is provided in
  the acceptance instructions).
- Upload the compiled `main.pdf` (or source zip) to obtain an
  IEEE Xplore-compatible certified PDF.
- Known local check (2026-07-20, pdffonts): all fonts embedded and subset
  (Type 1C + CID TrueType/Identity-H), no Type 3 fonts (matplotlib figure
  PDFs regenerated with fonttype 42).

## Post-submission

- [ ] Record the manuscript ID in the project ledger
  (`../paper2-uav/RUN_ON_36GB_MAC.md`).
- [ ] If/when posting this manuscript to arXiv (PI call), cross-list per the
  Paper 2 plan and add the arXiv ID to the ledger.
