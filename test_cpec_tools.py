"""Tests cpec_tools against a mock of jarvis' BoundingBox (no TensorFlow / jarvis needed)."""
import sys, types
import numpy as np

# stub tensorflow.keras so cpec_tools imports without TF
tf = types.ModuleType('tensorflow'); tf.keras = types.SimpleNamespace()
sys.modules['tensorflow'] = tf
sys.path.insert(0, sys.argv[1])
import matplotlib; matplotlib.use('Agg')
import cpec_tools as ct

G, S, A0 = 32, 8, 16.0          # grid, stride, anchor size
OFFSET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # 0 = jarvis-style grid (anchors at 0,8,...)
# Deliberately unusual channel order: [dx, dh, dy, dw] -> calibration must find x=0, y=2, h=1, w=3
IX, IH, IY, IW = 0, 1, 2, 3


class MockBB:
    cut = 0.5  # decoder compares raw cls value > 0.5 (unknown to cpec_tools)

    def centres(self):
        c = (np.arange(G) + OFFSET) * S
        return np.meshgrid(c, c, indexing='ij')  # cy, cx

    def convert_anc_to_box(self, boxes, classes):
        cls = np.zeros((1, 1, G, G, 1), np.float32); reg = np.zeros((1, 1, G, G, 4), np.float32)
        cy, cx = self.centres()
        anchors = np.stack([cy - A0/2, cx - A0/2, cy + A0/2, cx + A0/2], -1).reshape(-1, 4)
        if len(boxes):
            iou = ct.iou_matrix(anchors, boxes)
            for b_i, b in enumerate(boxes):
                pos = np.where(iou[:, b_i] >= 0.5)[0]
                if len(pos) == 0:
                    pos = [iou[:, b_i].argmax()]
                for a in pos:
                    r, c = divmod(a, G)
                    cls[0, 0, r, c, 0] = 1
                    by, bx = (b[0]+b[2])/2, (b[1]+b[3])/2
                    reg[0, 0, r, c, IY] = (by - cy[r, c]) / A0
                    reg[0, 0, r, c, IX] = (bx - cx[r, c]) / A0
                    reg[0, 0, r, c, IH] = np.log((b[2]-b[0]) / A0)
                    reg[0, 0, r, c, IW] = np.log((b[3]-b[1]) / A0)
        return {'cls-c3': cls, 'reg-c3': reg, 'cls-c3-msk': np.ones_like(cls), 'reg-c3-msk': np.ones_like(reg)}

    def convert_box_to_anc(self, d, iou_nms=0.5, apply_deltas=True):
        cy, cx = self.centres(); out = []
        for n in range(len(d['cls-c3'])):
            cls = np.asarray(d['cls-c3'][n]).reshape(G, G); reg = np.asarray(d['reg-c3'][n]).reshape(G, G, 4)
            rr, cc = np.where(cls > self.cut)
            boxes, scores = [], []
            for r, c in zip(rr, cc):
                y = cy[r, c] + reg[r, c, IY]*A0; x = cx[r, c] + reg[r, c, IX]*A0
                h = A0*np.exp(reg[r, c, IH]); w = A0*np.exp(reg[r, c, IW])
                boxes.append([y-h/2, x-w/2, y+h/2, x+w/2]); scores.append(cls[r, c])
            boxes = np.array(boxes).reshape(-1, 4); keep = []
            for i in np.argsort(-np.array(scores)):
                if all(ct.iou_matrix(boxes[i:i+1], boxes[k:k+1])[0, 0] < iou_nms for k in keep):
                    keep.append(i)
            out.append(boxes[keep])
        return out, None


rng = np.random.default_rng(0)
bb = MockBB()
N = 60
true_list, enc = [], []
for _ in range(N):
    k = rng.integers(0, 6)
    c = rng.uniform(20, 236, (k, 2)); hw = rng.uniform(12, 22, (k, 2))
    b = np.concatenate([c - hw/2, c + hw/2], 1)[:, [0, 1, 2, 3]]
    b = np.stack([c[:, 0]-hw[:, 0]/2, c[:, 1]-hw[:, 1]/2, c[:, 0]+hw[:, 0]/2, c[:, 1]+hw[:, 1]/2], 1)
    true_list.append(b); enc.append(bb.convert_anc_to_box(b, None))
data = {k: np.concatenate([e[k] for e in enc]) for k in enc[0]}
data['dat'] = rng.random((N, 256, 256, 3)).astype(np.float32)

# 1. ground-truth decode round-trips (NMS may merge heavily overlapping cells; allow small loss)
gt = ct.ground_truth_boxes(bb, data)
s, _ = ct.detection_metrics(gt, true_list, 0.5)
print('GT round trip:', s); assert s['recall'] > 0.95 and s['precision'] > 0.95

# 2. decoder cutoff discovered (mock uses raw > 0.5)
preds = {'cls-c3': np.where(data['cls-c3'] > 0, 2.0, -4.0).astype(np.float32), 'reg-c3': data['reg-c3']}
cut = ct.find_decoder_cutoff(bb, preds); print('cutoff', cut); assert abs(cut - 0.5) < 1e-3

# 3. threshold sweep: positives have logit 2 (p=.88), so thr .9 -> no detections, thr .5 -> all
sw = ct.threshold_sweep(bb, preds, gt, [0.5, 0.9], cutoff=cut); print(sw[['score_thr', 'tp', 'fp', 'fn', 'f1']])
assert sw.iloc[0].f1 > 0.99 and sw.iloc[1].tp == 0

# 4. metrics: a false positive and a missed cell are counted
p = [np.array([[10, 10, 26, 26], [100, 100, 116, 116]])]; t = [np.array([[11, 11, 27, 27], [200, 200, 216, 216]])]
s, _ = ct.detection_metrics(p, t); assert (s['tp'], s['fp'], s['fn']) == (1, 1, 1), s

# 5. geometry calibration finds the unusual layout and all three ops are exact
layout, ok = ct.calibrate_geometry(bb, data)
print(layout, ok)
assert layout['x'] == IX and layout['y'] == IY and layout['h'] == IH and layout['w'] == IW
assert set(ok) == {'hflip', 'vflip', 'transpose'}

# 6. a transformed sample's image and boxes stay aligned
smp = {k: v[3] for k, v in data.items()}
for ops in (['hflip'], ['vflip', 'transpose'], ['hflip', 'vflip', 'transpose']):
    t = ct.apply_geometry(smp, ops, layout)
    dec = ct.ground_truth_boxes(bb, {k: v[None] for k, v in t.items() if k != 'dat'})[0]
    exp = gt[3]
    for op in ops:
        exp = ct._mirror_boxes(exp, op, 256, 256)
    s_ = ct.detection_metrics([dec], [exp], 0.9)[0]
    assert s_['fp'] == 0 and s_['fn'] <= 1 and s_['tp'] >= len(exp) - 1, (ops, s_)
    # image pixel check: transform a marker pixel
    img = np.zeros((256, 256, 1)); img[5, 40] = 1
    im2 = ct.apply_geometry({'dat': img}, ops, layout)['dat']
    y, x = np.argwhere(im2[..., 0])[0]
    mk = ct._mirror_boxes(np.array([[5, 40, 6, 41]], float), ops[0], 256, 256)
    for op in ops[1:]:
        mk = ct._mirror_boxes(mk, op, 256, 256)
    assert (y, x) == (int(mk[0, 0]), int(mk[0, 1])), (ops, y, x, mk)

# 7. colour jitter survives per-tile normalization (unlike a single contrast factor)
x = rng.random((4, 64, 64, 3)).astype(np.float32)
def norm(a): a = a - a.mean((1, 2, 3), keepdims=True); return a / a.std((1, 2, 3), keepdims=True)
jit = ct.colour_jitter(x, np.random.default_rng(1))
contrast = (x - x.mean((1, 2), keepdims=True)) * 0.5 + x.mean((1, 2), keepdims=True)
print('normalized diff  jitter:', np.abs(norm(jit) - norm(x)).mean(), ' contrast:', np.abs(norm(contrast) - norm(x)).mean())
assert np.abs(norm(jit) - norm(x)).mean() > 5 * np.abs(norm(contrast) - norm(x)).mean()

# 8. box-size report flags boxes that 16px anchors cannot match
df = ct.box_size_report([np.array([[0, 0, 16, 16], [0, 0, 40, 40]])], plot=False)
assert df.best_centred_anchor_iou.round(3).tolist() == [1.0, 0.16]
print('ALL TESTS PASSED')
