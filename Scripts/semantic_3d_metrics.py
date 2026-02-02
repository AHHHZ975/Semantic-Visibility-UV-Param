
################################### Usage ##########################################
# python semantic_3d_metrics.py --gt samesh.glb --preds mine.glb blender.glb flexpara.glb --outdir out
####################################################################################


import os, argparse, json
import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
import pandas as pd

EPS = 1e-12

# ---------- helpers to load per-vertex labels from color ----------
def load_mesh_and_labels(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is None or len(vc) == 0:
        raise RuntimeError(f"No per-vertex colors in {path}")
    vc = np.asarray(vc)
    if vc.shape[1] >= 3:
        rgb = vc[:, :3].astype(np.uint8)
    else:
        raise RuntimeError("vertex_colors has <3 channels")
    tuples = [tuple(x.tolist()) for x in rgb]
    color2label = {}
    labels = np.zeros(len(tuples), dtype=np.int64)
    next_label = 0
    for i,t in enumerate(tuples):
        if t not in color2label:
            color2label[t] = next_label
            next_label += 1
        labels[i] = color2label[t]
    return mesh, labels, color2label

# ---------- permutation / mapping utilities ----------
def build_confusion(labelsA, labelsB):
    maxA = int(labelsA.max()) + 1
    maxB = int(labelsB.max()) + 1
    mat = np.zeros((maxA, maxB), dtype=np.int64)
    for a,b in zip(labelsA, labelsB):
        mat[int(a), int(b)] += 1
    return mat

def hungarian_map(conf):
    # maximize overlap -> minimize -conf
    cost = -conf.astype(np.int64)
    nrow, ncol = cost.shape
    n = max(nrow, ncol)
    pad = np.zeros((n,n), dtype=cost.dtype)
    pad[:nrow,:ncol] = cost
    rows, cols = linear_sum_assignment(pad)
    mapping = {}
    for r,c in zip(rows, cols):
        if r < nrow and c < ncol:
            mapping[int(c)] = int(r)
    return mapping

def map_labels(labels, mapping, default=-1):
    out = np.full_like(labels, default)
    for pred_label, ref_label in mapping.items():
        out[labels == pred_label] = ref_label
    return out

# ---------- metric implementations ----------
def hamming_after_perm(labels_gt, labels_pred):
    """
    Align pred -> gt using Hungarian mapping on the confusion matrix, then return
    (hamming_fraction, mapped_pred_labels, mapping dict).
    """
    conf = build_confusion(labels_gt, labels_pred)
    mapping = hungarian_map(conf)
    mapped = map_labels(labels_pred, mapping, default=-1)
    mismatch = (labels_gt != mapped)
    hamming_frac = float(mismatch.mean())
    return hamming_frac, mapped, mapping

def hamming_Rf_Rm(labels_gt, labels_pred):
    """
    Asymmetric Hamming style measures:
      - For each GT label g, find predicted label p* that maximizes overlap with g.
      - False positives for g: |P_p* \\ G_g|
      - False negatives for g: |G_g \\ P_p*|
    Aggregate all labels' FP and FN and normalize by total vertices.
    Returns: (Rf_frac, Rm_frac, per_label_details)
      - Rf_frac: total false positives fraction (lower is better)
      - Rm_frac: total false negatives fraction (lower is better)
    per_label_details: dict[label_g] -> {'best_pred': p_star, 'FP': count, 'FN': count, 'GT_size': count}
    """
    conf = build_confusion(labels_gt, labels_pred)  # rows GT, cols pred
    per_label = {}
    total_FP = 0
    total_FN = 0
    V = labels_gt.shape[0]
    for g in range(conf.shape[0]):
        row = conf[g, :]  # overlap of gt label g with each predicted label
        if row.sum() == 0:
            # this gt label has zero vertices? skip
            per_label[int(g)] = {'best_pred': -1, 'FP': 0, 'FN': 0, 'GT_size': 0}
            continue
        p_star = int(np.argmax(row))
        # size of predicted segment p_star
        pred_size = int(conf[:, p_star].sum())
        gt_size = int(row.sum())
        inter = int(row[p_star])
        FP = pred_size - inter          # predicted p_star vertices not in gt_g
        FN = gt_size - inter            # gt_g vertices not covered by pred p_star
        per_label[int(g)] = {'best_pred': int(p_star), 'FP': int(FP), 'FN': int(FN), 'GT_size': int(gt_size)}
        total_FP += FP
        total_FN += FN
    Rf = float(total_FP) / float(max(1, V))
    Rm = float(total_FN) / float(max(1, V))
    return Rf, Rm, per_label

def rand_index(labelsA, labelsB):
    """
    Fast Rand Index via contingency table counts.
    Returns RI in [0,1]. Higher is better.
    """
    n = labelsA.shape[0]
    if n < 2:
        return 1.0
    conf = build_confusion(labelsA, labelsB)
    nij = conf
    vec = nij.flatten()
    TP = float(np.sum(vec * (vec - 1) / 2.0))
    ai = np.sum(nij, axis=1)
    bj = np.sum(nij, axis=0)
    sum_ai = float(np.sum(ai * (ai - 1) / 2.0))
    sum_bj = float(np.sum(bj * (bj - 1) / 2.0))
    FP = sum_ai - TP
    FN = sum_bj - TP
    total_pairs = n * (n - 1) / 2.0
    TN = total_pairs - TP - FP - FN
    RI = (TP + TN) / (total_pairs + EPS)
    return float(RI)

def adjusted_rand(labelsA, labelsB):
    try:
        return float(adjusted_rand_score(labelsA, labelsB))
    except Exception:
        return float('nan')

def cut_discrepancy(meshA, labelsA, meshB, labelsB, sample_per_edge=1):
    """
    Approximate Cut Discrepancy between two segmentations:
      - find boundary edges (edges connecting vertices of different labels)
      - sample points (midpoints) on those edges (optionally subdivide edges)
      - compute mean nearest neighbour distance from setA -> setB and B -> A and sum them (symmetric)
    Returns (meanA2B, meanB2A, symmetric_sum)
    """
    # helper: get boundary edge midpoints
    def boundary_edge_points(mesh, labels, sample_per_edge=1):
        faces = np.asarray(mesh.faces)
        verts = np.asarray(mesh.vertices)
        edge_set = set()
        F = faces.shape[0]
        for f in faces:
            a,b,c = int(f[0]), int(f[1]), int(f[2])
            for (i,j) in ((a,b),(b,c),(c,a)):
                if labels[i] != labels[j]:
                    key = (min(i,j), max(i,j))
                    edge_set.add(key)
        pts = []
        for (i,j) in edge_set:
            p0 = verts[i]; p1 = verts[j]
            # sample midpoints or linear samples
            for s in range(sample_per_edge):
                t = (s + 0.5) / float(sample_per_edge)
                pts.append((1.0 - t) * p0 + t * p1)
        if len(pts) == 0:
            return np.zeros((0,3), dtype=float)
        return np.vstack(pts).astype(np.float64)

    ptsA = boundary_edge_points(meshA, labelsA, sample_per_edge=sample_per_edge)
    ptsB = boundary_edge_points(meshB, labelsB, sample_per_edge=sample_per_edge)
    if ptsA.shape[0] == 0 and ptsB.shape[0] == 0:
        return 0.0, 0.0, 0.0
    # kdtrees
    meanA2B = 0.0
    meanB2A = 0.0
    if ptsB.shape[0] > 0 and ptsA.shape[0] > 0:
        treeB = cKDTree(ptsB)
        distsA, _ = treeB.query(ptsA)
        meanA2B = float(np.mean(distsA))
        treeA = cKDTree(ptsA)
        distsB, _ = treeA.query(ptsB)
        meanB2A = float(np.mean(distsB))
    elif ptsA.shape[0] > 0:
        meanA2B = float(np.mean(np.linalg.norm(ptsA, axis=1)))  # fallback
        meanB2A = 0.0
    elif ptsB.shape[0] > 0:
        meanB2A = float(np.mean(np.linalg.norm(ptsB, axis=1)))
        meanA2B = 0.0
    return meanA2B, meanB2A, meanA2B + meanB2A

# ---------- high-level evaluation for one GT vs one predicted mesh ----------
def evaluate_one_pair(gt_mesh_path, pred_mesh_path, outdir=None, allow_remap=False):
    gt_mesh, gt_labels, _ = load_mesh_and_labels(gt_mesh_path)
    pred_mesh, pred_labels, _ = load_mesh_and_labels(pred_mesh_path)

    # if vertex counts differ, remap pred -> gt vertices by nearest neighbour if allowed
    if len(pred_labels) != len(gt_labels):
        if allow_remap:
            tree = cKDTree(np.asarray(pred_mesh.vertices))
            d, idx = tree.query(np.asarray(gt_mesh.vertices))
            pred_labels_mapped = pred_labels[idx]
        else:
            raise RuntimeError("Vertex count mismatch. Use allow_remap=True to project predicted labels to GT vertices.")
    else:
        pred_labels_mapped = pred_labels.copy()

    # Hamming (sym)
    hamming_frac, mapped_pred, mapping = hamming_after_perm(gt_labels, pred_labels_mapped)
    # Hamming Rf / Rm (asymmetric)
    Rf, Rm, per_label = hamming_Rf_Rm(gt_labels, pred_labels_mapped)
    # Rand / ARI
    RI = rand_index(gt_labels, pred_labels_mapped)
    ARI = adjusted_rand(gt_labels, pred_labels_mapped)
    # Cut discrepancy
    cA2B, cB2A, cut_sym = cut_discrepancy(gt_mesh, gt_labels, pred_mesh, pred_labels_mapped, sample_per_edge=1)

    results = {
        'gt': os.path.basename(gt_mesh_path),
        'pred': os.path.basename(pred_mesh_path),
        'hamming_frac': hamming_frac,
        'mapping': mapping,
        'Rf': Rf,
        'Rm': Rm,
        'rand_index': RI,
        'adjusted_rand': ARI,
        'cut_mean_gt2pred': cA2B,
        'cut_mean_pred2gt': cB2A,
        'cut_discrepancy_sym': cut_sym,
        'per_label_details': per_label
    }

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, f"eval_{os.path.basename(pred_mesh_path)}.json"), 'w') as fh:
            json.dump(results, fh, indent=2)
    return results

# ---------- CLI to evaluate a list of predicted methods vs SAMesh GT ----------
def evaluate_multiple(gt_path, pred_list, outdir, allow_remap=False):
    rows = []
    details = {}
    for p in pred_list:
        res = evaluate_one_pair(gt_path, p, outdir=outdir, allow_remap=allow_remap)
        rows.append({
            'pred': res['pred'],
            'hamming': res['hamming_frac'],
            'Rf': res['Rf'],
            'Rm': res['Rm'],
            'RI': res['rand_index'],
            'ARI': res['adjusted_rand'],
            'cut_sym': res['cut_discrepancy_sym']
        })
        details[res['pred']] = res
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'evaluation_summary.csv'), index=False)
    with open(os.path.join(outdir, 'evaluation_details.json'), 'w') as fh:
        json.dump(details, fh, indent=2)
    return df, details

# ---------- main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='SAMesh .glb (treated as GT)')
    parser.add_argument('--preds', nargs='+', required=True, help='Predicted method meshes (glb)')
    parser.add_argument('--outdir', required=True, help='Outdir to save results')
    parser.add_argument('--allow_remap', action='store_true', help='If vertex counts differ, remap pred->gt by nearest neighbor')
    args = parser.parse_args()
    df, details = evaluate_multiple(args.gt, args.preds, args.outdir, allow_remap=args.allow_remap)
    print("Summary saved to:", os.path.join(args.outdir, 'evaluation_summary.csv'))
    print(df)
