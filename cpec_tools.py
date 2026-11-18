"""
Tools for evaluating and improving the CPEC RetinaNet detector.

Contents
--------
1. Detection metrics      : precision / recall / F1 at an IoU cutoff, swept over score thresholds.
2. Diagnostics            : ground-truth box-size statistics vs. the anchor size, worst-tile viewer.
3. Augmentation           : a RetinaNet generator that flips/transposes images AND their box targets,
                            plus per-channel colour jitter that survives per-tile normalization.
4. Target re-encoding     : convert pickled datasets to a new BoundingBox (new anchor sizes/scales/ratios).

Box format everywhere is [y0, x0, y1, x1] in tile pixels (the format biondi uses).

Nothing here changes jarvis or biondi. Two things about jarvis are not assumed but measured at runtime:
  * the score cutoff used inside BoundingBox.convert_box_to_anc   -> find_decoder_cutoff()
  * how regression channels must change when a tile is flipped      -> calibrate_geometry()
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow import keras

try:  # biondi is only needed for the generator and re-encoding helpers
    import biondi
except ImportError:  # pragma: no cover
    biondi = None


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _as_boxes(a):
    """Decoder output for one tile -> float array of shape (n, 4)."""
    if a is None or (isinstance(a, str)):
        return np.zeros((0, 4), dtype=float)
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return np.zeros((0, 4), dtype=float)
    return a.reshape(-1, a.shape[-1])[:, :4]


def iou_matrix(a, b):
    """Pairwise IoU between boxes a (n,4) and b (m,4)."""
    a, b = _as_boxes(a), _as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    y0 = np.maximum(a[:, None, 0], b[None, :, 0])
    x0 = np.maximum(a[:, None, 1], b[None, :, 1])
    y1 = np.minimum(a[:, None, 2], b[None, :, 2])
    x1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(y1 - y0, 0, None) * np.clip(x1 - x0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_boxes(pred, true, iou_thr=0.5):
    """Greedy one-to-one matching by IoU. Returns (tp_pred_idx, tp_true_idx, fp_idx, fn_idx, matched_ious)."""
    pred, true = _as_boxes(pred), _as_boxes(true)
    ious = iou_matrix(pred, true)
    used_p, used_t, pairs = set(), set(), []
    if ious.size:
        order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
        for p, t in order:
            if ious[p, t] < iou_thr:
                break
            if p in used_p or t in used_t:
                continue
            used_p.add(p); used_t.add(t); pairs.append((p, t))
    tp_p = [p for p, _ in pairs]
    tp_t = [t for _, t in pairs]
    fp = [i for i in range(len(pred)) if i not in used_p]
    fn = [i for i in range(len(true)) if i not in used_t]
    return tp_p, tp_t, fp, fn, [ious[p, t] for p, t in pairs]


def _cls_keys(d):
    return [k for k in d if k.startswith('cls') and 'msk' not in k]


# ----------------------------------------------------------------------------------------------
# 1. predictions, decoding and metrics
# ----------------------------------------------------------------------------------------------
def predict_logits(model, data, batch_size=16, workers=8, max_queue_size=64, verbose=0):
    """Run the model once over a dataset dict and keep the raw output maps (so thresholds can be swept cheaply)."""
    # augmentation-free generator that always passes every input the model expects (older biondi drops cls/reg)
    gen = AugmentedRetinaNetGenerator(data=data, batch_size=batch_size, geo_ops=(), colour=False, validation=True)
    preds = model.predict(gen, workers=workers, max_queue_size=max_queue_size, verbose=verbose)
    if isinstance(preds, list):
        preds = {k: p for k, p in zip(model.output_names, preds)}
    return {k: np.asarray(v) for k, v in preds.items()}


def find_decoder_cutoff(bb, template, lo=-30.0, hi=30.0, iters=40):
    """
    Measure the value a cls output must exceed for bb.convert_box_to_anc to emit a box.
    `template` is any prediction dict (e.g. output of predict_logits); only its first tile is used.
    """
    base = {k: np.array(v[:1], dtype=np.float32) for k, v in template.items()}
    for k in base:
        base[k][:] = lo if k in _cls_keys(base) else 0.0
    ck = _cls_keys(base)[0]
    centre = tuple(s // 2 for s in base[ck].shape[1:-1])

    def emits(v):
        d = {k: x.copy() for k, x in base.items()}
        d[ck][(0,) + centre + (0,)] = v
        anc, _ = bb.convert_box_to_anc(d, iou_nms=0.3)
        return len(_as_boxes(anc[0])) > 0

    if not emits(hi):
        raise RuntimeError('Decoder never emitted a box; cannot calibrate threshold.')
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if emits(mid) else (mid, hi)
    return hi


def decode(bb, preds, score_thr=None, cutoff=None, iou_nms=0.3):
    """
    Decode prediction maps to boxes per tile. If score_thr (probability) is given, cls logits are shifted so the
    decoder keeps exactly the anchors with sigmoid(logit) > score_thr. `cutoff` comes from find_decoder_cutoff().
    """
    d = dict(preds)
    if score_thr is not None:
        if cutoff is None:
            raise ValueError('Pass cutoff=find_decoder_cutoff(bb, preds) when using score_thr.')
        shift = cutoff - _logit(score_thr)
        for k in _cls_keys(d):
            d[k] = d[k] + shift
    anc, _ = bb.convert_box_to_anc(d, iou_nms=iou_nms)
    return [_as_boxes(a) for a in anc]


def ground_truth_boxes(bb, data):
    """Decode the pickled targets to boxes, exactly as the original notebook does."""
    gt = {k: v for k, v in data.items() if k != 'dat'}
    anc, _ = bb.convert_box_to_anc(gt)
    return [_as_boxes(a) for a in anc]


def detection_metrics(pred_boxes, true_boxes, iou_thr=0.5):
    """Totals and per-tile counts for one set of predictions."""
    rows, tp, fp, fn, ious = [], 0, 0, 0, []
    for i, (p, t) in enumerate(zip(pred_boxes, true_boxes)):
        tp_p, _, f_p, f_n, m = match_boxes(p, t, iou_thr)
        rows.append(dict(tile=i, n_true=len(_as_boxes(t)), n_pred=len(_as_boxes(p)),
                         tp=len(tp_p), fp=len(f_p), fn=len(f_n)))
        tp += len(tp_p); fp += len(f_p); fn += len(f_n); ious += m
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    summary = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                   median_iou_of_matches=float(np.median(ious)) if ious else np.nan)
    return summary, pd.DataFrame(rows)


def threshold_sweep(bb, preds, true_boxes, thresholds=None, iou_thr=0.5, iou_nms=0.3, cutoff=None):
    """Precision/recall/F1 for each score threshold. Returns a DataFrame, one row per threshold."""
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if cutoff is None:
        cutoff = find_decoder_cutoff(bb, preds)
    out = []
    for thr in thresholds:
        s, _ = detection_metrics(decode(bb, preds, thr, cutoff, iou_nms), true_boxes, iou_thr)
        out.append(dict(score_thr=thr, **s))
    return pd.DataFrame(out)


def plot_sweep(df, title='Score threshold sweep'):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in ['precision', 'recall', 'f1']:
        ax[0].plot(df.score_thr, df[c], marker='o', label=c)
    ax[0].set_xlabel('score threshold'); ax[0].set_ylim(0, 1.02); ax[0].legend(); ax[0].set_title(title)
    ax[1].plot(df.recall, df.precision, marker='o')
    for _, r in df.iterrows():
        ax[1].annotate(f'{r.score_thr:g}', (r.recall, r.precision), fontsize=8)
    ax[1].set_xlabel('recall'); ax[1].set_ylabel('precision'); ax[1].set_title('Precision-recall')
    ax[1].set_xlim(0, 1.02); ax[1].set_ylim(0, 1.02)
    plt.tight_layout(); plt.show()


def evaluate_checkpoints(paths, bb, data, iou_thr=0.5, thresholds=None):
    """Best F1 (over score thresholds) for each saved model. Use to pick the checkpoint."""
    paths = list(paths)
    if not paths:
        raise ValueError('No checkpoints found. Check the folder path (use os.path.expanduser for "~") '
                         'and that it ends with "/".')
    true = ground_truth_boxes(bb, data)
    rows, cutoff = [], None
    for p in paths:
        model = keras.models.load_model(p)
        preds = predict_logits(model, data)
        cutoff = cutoff if cutoff is not None else find_decoder_cutoff(bb, preds)
        sw = threshold_sweep(bb, preds, true, thresholds, iou_thr, cutoff=cutoff)
        best = sw.loc[sw.f1.idxmax()]
        rows.append(dict(checkpoint=p, **best.to_dict()))
        print(p.split('/')[-1], f"F1={best.f1:.3f} P={best.precision:.3f} R={best.recall:.3f} @thr={best.score_thr}")
        keras.backend.clear_session()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 2. diagnostics
# ----------------------------------------------------------------------------------------------
def box_size_report(true_boxes, anchor_size=16, anchor_scales=(1,), anchor_ratios=(1,), iou_upper=0.5, plot=True):
    """
    Ground-truth box sizes and the best IoU any *centred* anchor could reach. A box whose best centred IoU is below
    iou_upper can never become a positive training example, so the model is taught to ignore it.
    (Real anchors also sit up to half a stride off-centre, so this is an optimistic upper bound.)
    """
    boxes = np.concatenate([_as_boxes(b) for b in true_boxes]) if len(true_boxes) else np.zeros((0, 4))
    h, w = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    best = np.zeros(len(boxes))
    for s in anchor_scales:
        for r in anchor_ratios:
            ah, aw = anchor_size * s * np.sqrt(r), anchor_size * s / np.sqrt(r)
            inter = np.minimum(h, ah) * np.minimum(w, aw)
            best = np.maximum(best, inter / (h * w + ah * aw - inter))
    df = pd.DataFrame(dict(height=h, width=w, best_centred_anchor_iou=best))
    print(f'{len(df)} boxes | height median {np.median(h):.1f} px (IQR {np.percentile(h,25):.1f}-{np.percentile(h,75):.1f})'
          f' | width median {np.median(w):.1f} px (IQR {np.percentile(w,25):.1f}-{np.percentile(w,75):.1f})')
    print(f'{(best < iou_upper).mean()*100:.1f}% of boxes cannot reach IoU {iou_upper} with any centred anchor')
    if plot and len(df):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].hist(h, 40, alpha=.6, label='height'); ax[0].hist(w, 40, alpha=.6, label='width')
        for s in anchor_scales:
            ax[0].axvline(anchor_size * s, color='k', ls='--', lw=1)
        ax[0].set_xlabel('px'); ax[0].legend(); ax[0].set_title('GT box size (dashed = anchor sizes)')
        ax[1].hist(best, 40); ax[1].axvline(iou_upper, color='r', ls='--')
        ax[1].set_title('Best centred-anchor IoU'); plt.tight_layout(); plt.show()
    return df


def show_tiles(data, pred_boxes, true_boxes, tiles, iou_thr=0.5, ncols=4, size=4):
    """Green = correct detection, red = false positive, yellow = missed CPEC (false negative)."""
    tiles = list(tiles)
    nrows = int(np.ceil(len(tiles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    imgs = np.asarray(data['dat'])
    imgs = imgs.reshape((-1,) + imgs.shape[-3:])  # -> (n, H, W, C) regardless of extra singleton axes
    for ax, i in zip(axes.ravel(), tiles):
        img = imgs[i][..., :3].astype(float)
        img = (img - img.min()) / (np.ptp(img) + 1e-9)
        ax.imshow(img)
        p, t = _as_boxes(pred_boxes[i]), _as_boxes(true_boxes[i])
        tp_p, _, fp, fn, _ = match_boxes(p, t, iou_thr)
        for idx, col in [(tp_p, 'lime'), (fp, 'red')]:
            for b in p[idx]:
                ax.add_patch(patches.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], fill=False, ec=col, lw=1.5))
        for b in t[fn]:
            ax.add_patch(patches.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], fill=False, ec='yellow',
                                           lw=1.5, ls='--'))
        ax.set_title(f'tile {i}: TP {len(tp_p)}  FP {len(fp)}  FN {len(fn)}', fontsize=9)
    plt.tight_layout(); plt.show()


def worst_tiles(per_tile, kind='fp', n=12):
    """Tile indices with the most false positives ('fp'), false negatives ('fn') or both ('err')."""
    df = per_tile.copy()
    df['err'] = df.fp + df.fn
    return df.sort_values(kind, ascending=False).head(n).tile.tolist()


# ----------------------------------------------------------------------------------------------
# 3. augmentation
# ----------------------------------------------------------------------------------------------
def _spatial_op(a, op):
    """Apply op to the two spatial axes (-3 = y, -2 = x) of an image or a target map."""
    if op == 'hflip':
        return a[..., :, ::-1, :]
    if op == 'vflip':
        return a[..., ::-1, :, :]
    if op == 'transpose':
        axes = list(range(a.ndim)); axes[-3], axes[-2] = axes[-2], axes[-3]
        return np.transpose(a, axes)
    raise ValueError(op)


def _reg_op(reg, op, layout):
    """Spatially transform a regression map, then fix the per-anchor deltas using the calibrated layout."""
    reg = _spatial_op(reg, op).copy()
    shp = reg.shape
    r = reg.reshape(shp[:-1] + (shp[-1] // 4, 4))
    if op == 'hflip':
        r[..., layout['x']] *= -1
    elif op == 'vflip':
        r[..., layout['y']] *= -1
    elif op == 'transpose':
        perm = np.arange(4)
        perm[[layout['y'], layout['x']]] = [layout['x'], layout['y']]
        perm[[layout['h'], layout['w']]] = [layout['w'], layout['h']]
        r[:] = r[..., perm]
    return r.reshape(shp)


def apply_geometry(sample, ops, layout):
    """sample: dict of arrays for ONE tile (dat + cls/reg/msk maps). Returns a transformed copy."""
    out = {}
    for k, v in sample.items():
        v = np.asarray(v)
        for op in ops:
            if k.startswith('reg') and 'msk' not in k:
                v = _reg_op(v, op, layout)
            else:
                v = _spatial_op(v, op)
        out[k] = np.ascontiguousarray(v)
    return out


def _mirror_boxes(b, op, H, W):
    b = _as_boxes(b).copy()
    if op == 'hflip':
        b[:, [1, 3]] = W - b[:, [3, 1]]
    elif op == 'vflip':
        b[:, [0, 2]] = H - b[:, [2, 0]]
    elif op == 'transpose':
        b = b[:, [1, 0, 3, 2]]
    return b


def calibrate_geometry(bb, data, n_tiles=40, tol=2.0, ops=('hflip', 'vflip', 'transpose')):
    """
    Work out which regression channel holds the y/x centre shifts and the h/w sizes, by flipping real ground-truth
    targets and checking that the decoded boxes land where the mirrored boxes should be.
    Returns (layout, ok_ops). Ops that cannot be made exact are left out of ok_ops (and so never used).
    """
    targets = {k: v for k, v in data.items() if k != 'dat'}
    H, W = np.squeeze(np.asarray(data['dat'][0])).shape[:2]
    idx = [i for i in range(len(data['dat'])) if len(ground_truth_boxes(bb, {k: v[i:i+1] for k, v in targets.items()})[0])]
    idx = idx[:n_tiles]
    if not idx:
        raise RuntimeError('No tiles with boxes found for calibration.')
    sub = {k: np.asarray(v)[idx] for k, v in targets.items()}
    ref = ground_truth_boxes(bb, sub)

    def max_err(op, layout):
        moved = {k: np.stack([apply_geometry({k: v[i]}, [op], layout)[k] for i in range(len(idx))])
                 for k, v in sub.items()}
        dec = ground_truth_boxes(bb, moved)
        errs, n_total, n_unmatched = [], 0, 0
        for d, r in zip(dec, ref):
            m = _mirror_boxes(r, op, H, W)
            tp_p, tp_t, fp, fn, _ = match_boxes(d, m, 0.3)
            n_total += len(m); n_unmatched += len(fn) + len(fp)
            errs += [np.abs(d[p] - m[t]).max() for p, t in zip(tp_p, tp_t)]
        # NMS tie-breaking can legitimately change which of two overlapping boxes survives, so judge on
        # the typical (90th percentile) error and require almost every box to be matched.
        if not errs or n_unmatched > 0.05 * max(n_total, 1):
            return np.inf
        return float(np.percentile(errs, 90))

    layout, ok = {}, []
    for axis, op in [('x', 'hflip'), ('y', 'vflip')]:
        errs = {c: max_err(op, {'x': c, 'y': c, 'h': 0, 'w': 0}) for c in range(4)}
        c = min(errs, key=errs.get)
        layout[axis] = c
        if errs[c] <= tol and op in ops:
            ok.append(op)
        print(f'{op}: channel {c} max error {errs[c]:.2f} px', '(ok)' if errs[c] <= tol else '(DISABLED)')
    rest = [c for c in range(4) if c not in (layout['x'], layout['y'])]
    if 'transpose' in ops and len(rest) == 2 and layout['x'] != layout['y']:
        best = None
        for h, w in [(rest[0], rest[1]), (rest[1], rest[0])]:
            e = max_err('transpose', dict(layout, h=h, w=w))
            if best is None or e < best[0]:
                best = (e, h, w)
        layout['h'], layout['w'] = best[1], best[2]
        if best[0] <= tol:
            ok.append('transpose')
        print(f'transpose: max error {best[0]:.2f} px', '(ok)' if best[0] <= tol else '(DISABLED)')
    else:
        layout.setdefault('h', rest[0] if rest else 2); layout.setdefault('w', rest[-1] if rest else 3)
    return layout, ok


def colour_jitter(x, rng, gain=0.12, bias=0.06, sat=0.2):
    """
    Per-image, per-channel gain/offset + saturation jitter on raw RGB tiles (N,H,W,3).
    Unlike a single contrast factor, per-channel changes are NOT removed by per-tile z-score normalization,
    so they actually simulate stain/scanner variation.
    """
    x = x.astype(np.float32)
    n, c = x.shape[0], x.shape[-1]
    scale = x.reshape(n, -1).std(1).reshape(n, 1, 1, 1) + 1e-6
    g = 1 + rng.uniform(-gain, gain, (n, 1, 1, c))
    b = rng.uniform(-bias, bias, (n, 1, 1, c)) * scale
    x = x * g + b
    if c >= 3 and sat:
        grey = x[..., :3].mean(-1, keepdims=True)
        s = 1 + rng.uniform(-sat, sat, (n, 1, 1, 1))
        x[..., :3] = grey + (x[..., :3] - grey) * s
    return x


if biondi is not None:
    class AugmentedRetinaNetGenerator(biondi.dataset.TrainingGenerator):
        """
        Drop-in replacement for TrainingGenerator(retinanet=True) for TRAINING data.
          geo_ops : subset of calibrate_geometry's ok_ops; each op is applied with p=0.5 per tile
                    (hflip+vflip+transpose together give all 8 flips/90-degree rotations).
          colour  : per-channel colour jitter (see colour_jitter).
        Pass layout/geo_ops from calibrate_geometry(bb, x). Validation/test generators should stay plain.
        """
        def __init__(self, data, batch_size, layout=None, geo_ops=(), colour=True, seed=None, **kwargs):
            import inspect
            accepted = inspect.signature(biondi.dataset.TrainingGenerator.__init__).parameters
            kwargs.setdefault('retinanet', True)
            # older biondi versions don't have every augmentation option; only pass the ones this version takes
            for opt in ('contrast', 'flip', 'rotation'):
                if opt in accepted:
                    kwargs[opt] = False
                else:
                    kwargs.pop(opt, None)
            super().__init__(data=data, batch_size=batch_size, **kwargs)
            self.normalize = getattr(self, 'normalize', kwargs.get('normalize', True))
            self.per_channel = getattr(self, 'per_channel', kwargs.get('per_channel', False))
            self.c_idx_start = getattr(self, 'c_idx_start', 0)
            if not hasattr(self, 'keys'):
                self.keys = self.data.keys()
            if not hasattr(self, 'indexes'):
                self.indexes = np.arange(len(self.data['dat']))
            self.layout, self.geo_ops, self.colour = layout, tuple(geo_ops), colour
            self.rng = np.random.default_rng(seed)
            if self.geo_ops and layout is None:
                raise ValueError('geo_ops need a layout from calibrate_geometry().')

        def __getitem__(self, index):
            batch_idx = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            batch = {k: np.asarray(self.data[k][batch_idx]) for k in self.keys}
            batch['dat'] = batch['dat'][..., self.c_idx_start:]
            # always work on (n, H, W, C), whatever extra singleton axes the stored data has
            batch['dat'] = batch['dat'].reshape((len(batch_idx),) + batch['dat'].shape[-3:])
            if self.geo_ops:
                for j in range(len(batch_idx)):
                    ops = [op for op in self.geo_ops if self.rng.random() < 0.5]
                    if ops:
                        t = apply_geometry({k: v[j] for k, v in batch.items()}, ops, self.layout)
                        for k in batch:
                            batch[k][j] = t[k]
            if self.colour:
                batch['dat'] = colour_jitter(batch['dat'], self.rng)
            if self.normalize:
                batch['dat'] = biondi.dataset.per_sample_tile_normalization(batch['dat'], per_channel=self.per_channel)
            batch['dat'] = np.expand_dims(batch['dat'], axis=1)
            return batch


# ----------------------------------------------------------------------------------------------
# 4. re-encoding targets for a new anchor configuration
# ----------------------------------------------------------------------------------------------
def reencode_targets(data, old_bb, new_bb):
    """
    Decode the pickled targets with the BoundingBox they were made with and re-encode them with a new one
    (e.g. more anchor scales/ratios). Returns a new dataset dict with the same 'dat'.
    Tiles without boxes get all-negative targets from new_bb.
    """
    true = ground_truth_boxes(old_bb, data)
    out = {}
    for i, b in enumerate(true):
        b = _as_boxes(b)
        enc = new_bb.convert_anc_to_box(b, np.ones((b.shape[0], 1)))
        for k, v in enc.items():
            out.setdefault(k, []).append(np.asarray(v))
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{len(true)}', end='\r')
    new = {k: np.concatenate(v, 0) for k, v in out.items()}
    new['dat'] = data['dat']
    for k, v in new.items():
        print(k, v.shape)
    return new