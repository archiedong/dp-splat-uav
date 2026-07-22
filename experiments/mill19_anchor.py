#!/usr/bin/env python
"""Mill 19 "Rubble" COLMAP sparse metric anchor (Phase B-3 prerequisite).

Converts the Mega-NeRF-format release (rubble-pixsfm) back into a metric
COLMAP model with FIXED camera poses, then triangulates a colored sparse
point cloud from CPU SIFT features as the geometric anchor for the
re-flight experiment.

Mega-NeRF conventions (verified 2026-07-16 against
github.com/cmusatyalab/mega-nerf, scripts/colmap_to_mega_nerf.py and
mega_nerf/ray_utils.py):

  Forward transform (COLMAP -> stored metadata), quoted from their script:
      w2c = [R(qvec) | tvec];  c2w = inv(w2c)            # OpenCV cam -> COLMAP world
      A   = RDF_TO_DRB = [[0,1,0],[1,0,0],[0,0,-1]]      # involution, A == A^-1, det=+1
      c2w_drb = [ A @ c2w[:3,:3] @ A^-1 | A @ c2w[:3,3] ]
      c2w_drb[:,3] = (c2w_drb[:,3] - origin_drb) / pose_scale_factor
      stored  = cat([c2w_drb[:,1:2], -c2w_drb[:,0:1], c2w_drb[:,2:4]], -1)

  Ray convention (ray_utils.py):
      dirs_cam = [(i - cx)/fx, -(j - cy)/fy, -1]  (OpenGL cam: x right, y up, z back)
      rays_d   = dirs_cam @ stored[:,:3].T
  i.e. stored[:,:3] maps OpenGL-style camera axes into the DRB world
  (x down, y right, z back); translations are normalized by
  pose_scale_factor about origin_drb (coordinates.pt).

  This script applies the EXACT inverse (and self-tests it by re-running
  the forward transform on the recovered poses):
      c2w_drb[:,0] = -stored[:,1]; c2w_drb[:,1] = stored[:,0]
      c2w_drb[:,2:4] = stored[:,2:4]
      t_metric = stored_t * pose_scale_factor + origin_drb
      R_c2w = A @ R_drb @ A ;  t_c2w = A @ t_metric      # back to COLMAP world, metric
      w2c   = [R_c2w^T | -R_c2w^T t_c2w]  -> qvec/tvec for images.txt

  Recovered COLMAP world axes: x = right, y = down (vertical), z = forward.
  Horizontal plane = (x, z).

Intrinsics: all 1678 images share fx=fy=2977.5291, cx=2304, cy=1728,
distortion [k1=-0.00197067, 0, 0, 0] at 4608x3456. At the 4x-downscaled
working resolution (1152x864) intrinsics scale linearly by 0.25 (COLMAP
continuous pixel convention); k1 is on normalized coords and is invariant.
Because intrinsics are identical, ONE shared camera is written instead of
per-image copies (strictly better-constrained, same geometry). Both an
OPENCV (with k1) and a PINHOLE variant are triangulated; the lower
mean-reprojection-error variant is exported (decides empirically whether
the released rgbs are distorted originals or already undistorted).

Matching choice: vocab-tree retrieval is unavailable offline, so we use
sequential matching (images are in capture order: global 6-digit ids from
mappings.txt) PLUS a pose-prior spatial pair list (k nearest cameras by
metric 3D distance) imported via `colmap matches_importer`. This covers
both along-track and cross-track (adjacent lawnmower strip) overlap.

Usage:
    python experiments/mill19_anchor.py [--stages all]
    stages: convert,downscale,extract,match,triangulate,export,report
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# paths / constants
# ----------------------------------------------------------------------------
DATA = Path.home() / "dp-splat-data/mill19/rubble-pixsfm"
WORK = Path.home() / "dp-splat-data/mill19/colmap_work"
IMG_DIR = WORK / "images_4"
DB_PATH = WORK / "database.db"
PAIRS_PATH = WORK / "spatial_pairs.txt"
PLY_PATH = WORK / "mill19_anchor_points.ply"
REPO = Path(__file__).resolve().parent
OUT = REPO / "out"
REPORT_PATH = OUT / "mill19_anchor_report.json"
TIMES_PATH = WORK / "stage_runtimes.json"

DOWNSCALE = 4
FULL_W, FULL_H = 4608, 3456
W4, H4 = FULL_W // DOWNSCALE, FULL_H // DOWNSCALE
N_THREADS = 8          # cap: HEAVY fits are sharing this machine
NICENESS = 15
K_SPATIAL = 24         # spatial pair neighbors per image
SEQ_OVERLAP = 10

A_RDF_DRB = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]])  # involution: A @ A = I


def be_nice() -> None:
    cur = os.nice(0)
    if cur < NICENESS:
        os.nice(NICENESS - cur)


# ----------------------------------------------------------------------------
# quaternion helpers (COLMAP convention: [qw qx qy qz], Hamilton)
# ----------------------------------------------------------------------------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    return -q if q[0] < 0 else q


# ----------------------------------------------------------------------------
# metadata loading + pose conversion
# ----------------------------------------------------------------------------
def load_all():
    import torch
    coords = torch.load(DATA / "coordinates.pt", map_location="cpu",
                        weights_only=False)
    origin = coords["origin_drb"].double().numpy()
    psf = float(coords["pose_scale_factor"])
    recs = []
    for split in ("train", "val"):
        for p in sorted((DATA / split / "metadata").glob("*.pt")):
            gid = int(p.stem)
            m = torch.load(p, map_location="cpu", weights_only=False)
            rgb = DATA / split / "rgbs" / f"{gid:06d}.jpg"
            if not rgb.exists():
                raise FileNotFoundError(rgb)
            recs.append(dict(
                gid=gid, split=split, rgb=rgb,
                c2w=m["c2w"].double().numpy(),
                intrinsics=m["intrinsics"].double().numpy(),
                distortion=m["distortion"].double().numpy(),
                W=int(m["W"]), H=int(m["H"]),
            ))
    recs.sort(key=lambda r: r["gid"])
    assert [r["gid"] for r in recs] == list(range(len(recs))), "non-contiguous ids"
    return recs, origin, psf


def meganerf_to_colmap(stored_c2w, origin, psf):
    """Invert the mega-nerf forward transform. Returns (qvec, tvec, C_metric)
    where qvec/tvec are the COLMAP world->cam pose and C_metric the camera
    center in the metric COLMAP-world frame (x right, y down, z forward)."""
    A = A_RDF_DRB
    stored = np.asarray(stored_c2w, dtype=np.float64)
    R_drb = np.column_stack([-stored[:, 1], stored[:, 0], stored[:, 2]])
    t_drb = stored[:, 3] * psf + origin
    R_c2w = A @ R_drb @ A
    t_c2w = A @ t_drb
    # project to SO(3) (metadata is float32)
    U, _, Vt = np.linalg.svd(R_c2w)
    R_c2w = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    R_w2c = R_c2w.T
    tvec = -R_w2c @ t_c2w
    return rotmat2qvec(R_w2c), tvec, t_c2w


def colmap_to_meganerf(qvec, tvec, origin, psf):
    """Mega-NeRF forward transform, verbatim math, for round-trip self-test."""
    A = A_RDF_DRB
    R_w2c = qvec2rotmat(qvec)
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ tvec
    R_drb = A @ R_c2w @ np.linalg.inv(A)
    t_drb = A @ t_c2w
    t_norm = (t_drb - origin) / psf
    drb = np.column_stack([R_drb, t_norm])
    return np.column_stack([drb[:, 1:2], -drb[:, 0:1], drb[:, 2:4]])


def stage_convert(recs, origin, psf):
    """Convert all poses, self-test round trip, cache results."""
    poses = {}
    max_err = 0.0
    for r in recs:
        qvec, tvec, C = meganerf_to_colmap(r["c2w"], origin, psf)
        rt = colmap_to_meganerf(qvec, tvec, origin, psf)
        err = np.abs(rt - r["c2w"]).max()
        max_err = max(max_err, err)
        poses[r["gid"]] = dict(qvec=qvec.tolist(), tvec=tvec.tolist(),
                               C=C.tolist(), split=r["split"])
    assert max_err < 1e-3, f"round-trip failed: {max_err}"
    print(f"[convert] {len(poses)} poses, round-trip max |err| = {max_err:.2e}")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "poses_colmap.json").write_text(json.dumps(
        dict(origin_drb=origin.tolist(), pose_scale_factor=psf, poses=poses)))
    return poses


# ----------------------------------------------------------------------------
# image downscaling
# ----------------------------------------------------------------------------
def _downscale_one(args):
    src, dst = args
    from PIL import Image
    if Path(dst).exists():
        return False
    im = Image.open(src)
    im = im.resize((W4, H4), Image.LANCZOS)
    im.save(str(dst) + ".part", format="JPEG", quality=95)
    os.replace(str(dst) + ".part", dst)
    return True


def stage_downscale(recs):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(str(r["rgb"]), str(IMG_DIR / f"{r['gid']:06d}.jpg")) for r in recs]
    t0 = time.time()
    with Pool(N_THREADS, initializer=be_nice) as pool:
        done = sum(pool.imap_unordered(_downscale_one, jobs, chunksize=16))
    print(f"[downscale] {done} new / {len(jobs)} total "
          f"({time.time() - t0:.1f}s) -> {IMG_DIR}")


# ----------------------------------------------------------------------------
# COLMAP driving
# ----------------------------------------------------------------------------
def run_colmap(args, log_name):
    WORK.mkdir(parents=True, exist_ok=True)
    log = WORK / f"log_{log_name}.txt"
    cmd = ["colmap"] + [str(a) for a in args]
    print(f"[colmap] {' '.join(cmd[:2])} ... (log: {log.name})")
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    _record_time(log_name, dt)
    print(f"[colmap] {log_name} done in {dt:.1f}s")
    return dt


def _record_time(key, dt):
    times = json.loads(TIMES_PATH.read_text()) if TIMES_PATH.exists() else {}
    times[key] = round(dt, 1)
    TIMES_PATH.write_text(json.dumps(times, indent=1))


def camera_params_str(model):
    fx = 2977.529052734375 / DOWNSCALE   # exact float64 promotion of metadata
    cx, cy = 2304.0 / DOWNSCALE, 1728.0 / DOWNSCALE
    k1 = -0.0019706706516444683
    if model == "OPENCV":
        p = [fx, fx, cx, cy, k1, 0.0, 0.0, 0.0]
    elif model == "PINHOLE":
        p = [fx, fx, cx, cy]
    else:
        raise ValueError(model)
    return ",".join(f"{v:.10g}" for v in p)


def stage_extract():
    run_colmap([
        "feature_extractor",
        "--database_path", DB_PATH,
        "--image_path", IMG_DIR,
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", 1,
        "--ImageReader.camera_params", camera_params_str("OPENCV"),
        "--FeatureExtraction.max_image_size", 1600,
        "--FeatureExtraction.use_gpu", 0,
        "--FeatureExtraction.num_threads", N_THREADS,
    ], "feature_extractor")


def db_image_ids():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    con.close()
    return {name: (img_id, cam_id) for img_id, name, cam_id in rows}


def write_pairs(poses):
    from scipy.spatial import cKDTree
    gids = sorted(poses)
    C = np.array([poses[g]["C"] for g in gids])
    tree = cKDTree(C)
    _, nn = tree.query(C, k=K_SPATIAL + 1)
    pairs = set()
    for i, row in enumerate(nn):
        for j in row[1:]:
            if abs(gids[i] - gids[int(j)]) <= SEQ_OVERLAP:
                continue  # sequential matcher already covers these
            pairs.add((min(i, int(j)), max(i, int(j))))
    with open(PAIRS_PATH, "w") as f:
        for i, j in sorted(pairs):
            f.write(f"{gids[i]:06d}.jpg {gids[j]:06d}.jpg\n")
    print(f"[pairs] {len(pairs)} spatial pairs (k={K_SPATIAL}) -> {PAIRS_PATH}")


def stage_match(poses):
    run_colmap([
        "sequential_matcher",
        "--database_path", DB_PATH,
        "--SequentialMatching.overlap", SEQ_OVERLAP,
        "--SequentialMatching.loop_detection", 0,
        "--FeatureMatching.use_gpu", 0,
        "--FeatureMatching.num_threads", N_THREADS,
    ], "sequential_matcher")
    write_pairs(poses)
    run_colmap([
        "matches_importer",
        "--database_path", DB_PATH,
        "--match_list_path", PAIRS_PATH,
        "--match_type", "pairs",
        "--FeatureMatching.use_gpu", 0,
        "--FeatureMatching.num_threads", N_THREADS,
    ], "matches_importer_spatial")


def write_text_model(path, poses, camera_model):
    path.mkdir(parents=True, exist_ok=True)
    by_name = db_image_ids()
    cam_ids = {cam_id for (_, cam_id) in by_name.values()}
    assert len(cam_ids) == 1, f"expected 1 shared camera, got {len(cam_ids)}"
    cam_id = cam_ids.pop()
    params = camera_params_str(camera_model).replace(",", " ")
    with open(path / "cameras.txt", "w") as f:
        f.write("# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        f.write(f"{cam_id} {camera_model} {W4} {H4} {params}\n")
    with open(path / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for gid in sorted(poses):
            name = f"{gid:06d}.jpg"
            if name not in by_name:
                raise KeyError(f"{name} missing from database")
            img_id, _ = by_name[name]
            q = poses[gid]["qvec"]
            t = poses[gid]["tvec"]
            f.write(f"{img_id} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
                    f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {cam_id} {name}\n\n")
    (path / "points3D.txt").write_text("")
    print(f"[model] wrote fixed-pose text model ({camera_model}) -> {path}")


def stage_triangulate(poses):
    stats = {}
    for model in ("OPENCV", "PINHOLE"):
        tag = model.lower()
        sparse_in = WORK / f"sparse_in_{tag}"
        sparse_out = WORK / f"sparse_{tag}"
        write_text_model(sparse_in, poses, model)
        sparse_out.mkdir(parents=True, exist_ok=True)
        run_colmap([
            "point_triangulator",
            "--database_path", DB_PATH,
            "--image_path", IMG_DIR,
            "--input_path", sparse_in,
            "--output_path", sparse_out,
            "--clear_points", 1,
            "--refine_intrinsics", 0,
            "--Mapper.num_threads", N_THREADS,
        ], f"point_triangulator_{tag}")
        run_colmap([
            "model_converter",
            "--input_path", sparse_out,
            "--output_path", sparse_out,
            "--output_type", "TXT",
        ], f"model_converter_txt_{tag}")
        stats[model] = summarize_points(sparse_out / "points3D.txt")
        print(f"[triangulate:{model}] {stats[model]['n_points']} pts, "
              f"mean reproj {stats[model]['mean_reproj_error_px']:.3f}px, "
              f"mean track {stats[model]['mean_track_length']:.2f}")
    (WORK / "triangulation_ab.json").write_text(json.dumps(stats, indent=1))
    return stats


def summarize_points(points3d_txt):
    n = 0
    err_sum = 0.0
    track_sum = 0
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    xs = []
    with open(points3d_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            err = float(parts[7])
            track = (len(parts) - 8) // 2
            n += 1
            err_sum += err
            track_sum += track
            mins = np.minimum(mins, xyz)
            maxs = np.maximum(maxs, xyz)
            xs.append(xyz)
    xs = np.array(xs) if xs else np.zeros((0, 3))
    if n:
        lo = np.percentile(xs, 0.5, axis=0)
        hi = np.percentile(xs, 99.5, axis=0)
    else:
        lo = hi = np.zeros(3)
    return dict(
        n_points=n,
        mean_reproj_error_px=err_sum / max(n, 1),
        mean_track_length=track_sum / max(n, 1),
        bbox_min=mins.tolist(), bbox_max=maxs.tolist(),
        robust_bbox_min=lo.tolist(), robust_bbox_max=hi.tolist(),
        robust_extent_m=(hi - lo).tolist(),
    )


def stage_export(winner_tag):
    sparse_out = WORK / f"sparse_{winner_tag}"
    final = WORK / "sparse_final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(sparse_out, final)
    run_colmap([
        "model_converter",
        "--input_path", final,
        "--output_path", PLY_PATH,
        "--output_type", "PLY",
    ], "model_converter_ply")
    print(f"[export] final model ({winner_tag}) -> {final}\n[export] PLY -> {PLY_PATH}")


# ----------------------------------------------------------------------------
# verification + report
# ----------------------------------------------------------------------------
def lawnmower_stats(poses):
    gids = sorted(poses)
    C = np.array([poses[g]["C"] for g in gids])  # x right, y down, z forward
    horiz = C[:, [0, 2]]
    v = np.diff(horiz, axis=0)
    norms = np.linalg.norm(v, axis=1)
    keep = norms > 1e-6
    u = v[keep] / norms[keep][:, None]
    dots = (u[:-1] * u[1:]).sum(1)
    reversals = int((dots < -0.7).sum())
    return dict(
        n_cameras=len(C),
        altitude_axis="y (down)",
        altitude_span_m=float(C[:, 1].max() - C[:, 1].min()),
        horiz_extent_m=[float(horiz[:, 0].max() - horiz[:, 0].min()),
                        float(horiz[:, 1].max() - horiz[:, 1].min())],
        horiz_bbox_area_m2=float((horiz[:, 0].max() - horiz[:, 0].min())
                                 * (horiz[:, 1].max() - horiz[:, 1].min())),
        heading_reversals=reversals,
        median_step_m=float(np.median(norms[keep])),
    )


def camera_figure(poses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gids = sorted(poses)
    C = np.array([poses[g]["C"] for g in gids])
    val = np.array([poses[g]["split"] == "val" for g in gids])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(C[:, 0], C[:, 2], "-", lw=0.4, color="0.7", zorder=1)
    sc = ax.scatter(C[:, 0], C[:, 2], c=gids, s=4, cmap="viridis", zorder=2)
    ax.scatter(C[val, 0], C[val, 2], s=30, facecolors="none",
               edgecolors="red", lw=0.8, zorder=3, label="val")
    ax.set_xlabel("x (m, right)")
    ax.set_ylabel("z (m, forward)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.colorbar(sc, ax=ax, label="capture order (global id)")
    ax.set_title("Mill 19 Rubble: camera positions (metric, COLMAP world)")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mill19_anchor_cameras.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[report] camera figure -> {OUT / 'mill19_anchor_cameras.png'}")


def stage_report(poses, ab_stats, winner, psf, origin):
    lawn = lawnmower_stats(poses)
    camera_figure(poses)
    final_stats = ab_stats[winner]
    times = json.loads(TIMES_PATH.read_text()) if TIMES_PATH.exists() else {}
    ext = final_stats["robust_extent_m"]
    checks = dict(
        n_points_gt_1e5=final_stats["n_points"] > 1e5,
        horiz_extent_plausible_for_110000m2_site=(
            100.0 < ext[0] < 1000.0 and 100.0 < ext[2] < 1000.0),
        lawnmower_pattern=(lawn["heading_reversals"] >= 8
                           and lawn["altitude_span_m"] < 30.0),
        round_trip_pose_selftest="passed in convert stage",
    )
    report = dict(
        dataset="mill19/rubble-pixsfm (Mega-NeRF release)",
        n_images=lawn["n_cameras"],
        n_train=sum(1 for p in poses.values() if p["split"] == "train"),
        n_val=sum(1 for p in poses.values() if p["split"] == "val"),
        downscale=DOWNSCALE, work_res=[W4, H4],
        pose_scale_factor=psf, origin_drb=origin.tolist(),
        camera_model_final=winner,
        camera_model_ab={m: dict(n_points=s["n_points"],
                                 mean_reproj_error_px=round(s["mean_reproj_error_px"], 4))
                         for m, s in ab_stats.items()},
        n_points=final_stats["n_points"],
        mean_reproj_error_px=round(final_stats["mean_reproj_error_px"], 4),
        mean_track_length=round(final_stats["mean_track_length"], 2),
        point_cloud_robust_extent_m=[round(e, 1) for e in ext],
        point_cloud_robust_bbox_min=[round(v, 1) for v in final_stats["robust_bbox_min"]],
        point_cloud_robust_bbox_max=[round(v, 1) for v in final_stats["robust_bbox_max"]],
        world_frame="metric COLMAP world: x right, y down (vertical), z forward",
        cameras=lawn,
        checks=checks,
        stage_runtimes_s=times,
        paths=dict(ply=str(PLY_PATH), model=str(WORK / "sparse_final"),
                   images=str(IMG_DIR), database=str(DB_PATH)),
        conventions=("Mega-NeRF DRB (x down, y right, z back), "
                     "stored c2w maps OpenGL cam axes -> normalized DRB world; "
                     "inverse verified by round-trip vs "
                     "scripts/colmap_to_mega_nerf.py conventions"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1))
    print(f"[report] -> {REPORT_PATH}")
    print(json.dumps(checks, indent=1))
    return report


# ----------------------------------------------------------------------------
def main():
    be_nice()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all",
                    help="comma list: convert,downscale,extract,match,"
                         "triangulate,export,report (default all)")
    args = ap.parse_args()
    stages = (["convert", "downscale", "extract", "match", "triangulate",
               "export", "report"] if args.stages == "all"
              else args.stages.split(","))

    recs, origin, psf = load_all()
    t_all = time.time()

    if "convert" in stages or not (WORK / "poses_colmap.json").exists():
        t0 = time.time()
        poses = stage_convert(recs, origin, psf)
        _record_time("convert", time.time() - t0)
    else:
        cached = json.loads((WORK / "poses_colmap.json").read_text())
        poses = {int(k): v for k, v in cached["poses"].items()}

    if "downscale" in stages:
        t0 = time.time()
        stage_downscale(recs)
        _record_time("downscale", time.time() - t0)
    if "extract" in stages:
        stage_extract()
    if "match" in stages:
        stage_match(poses)

    ab_stats = None
    if "triangulate" in stages:
        ab_stats = stage_triangulate(poses)
    if ab_stats is None and (WORK / "triangulation_ab.json").exists():
        ab_stats = json.loads((WORK / "triangulation_ab.json").read_text())

    winner = None
    if ab_stats:
        winner = min(ab_stats, key=lambda m: (
            ab_stats[m]["mean_reproj_error_px"]
            if ab_stats[m]["n_points"] > 1e5 else np.inf))
        print(f"[ab] winner: {winner} "
              f"({ {m: round(s['mean_reproj_error_px'], 4) for m, s in ab_stats.items()} })")

    if "export" in stages and winner:
        stage_export(winner.lower())
    if "report" in stages and winner:
        _record_time("total_this_run", time.time() - t_all)
        stage_report(poses, ab_stats, winner, psf, origin)


if __name__ == "__main__":
    main()
