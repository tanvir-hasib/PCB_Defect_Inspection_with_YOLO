"""Shared inference core for PCB defect inspection (numpy + OpenCV only).

Used by BOTH the notebook and the FastAPI server, so training-time evaluation and
production inference run exactly the same pre-processing, tiling, decoding and merging.

Model contract (Ultralytics YOLOv8/YOLO11 detection head, exported to ONNX):
    input  "images"  float32 [B, 3, S, S]  RGB, 0..1
    output "output0" float32 [B, 4 + nc, A]  (cx, cy, w, h in input pixels, then per-class scores)
End-to-end heads (e.g. YOLO26) that output [B, max_det, 6] = (x1, y1, x2, y2, score, class) are also handled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

CLASSES = ("missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper")

RunFn = Callable[[np.ndarray], np.ndarray]   # float32 NCHW blob -> raw model output


@dataclass
class Detections:
    boxes: np.ndarray    # [N, 4] x1, y1, x2, y2 in original image pixels
    scores: np.ndarray   # [N]
    classes: np.ndarray  # [N] int

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.scores)

    def filter(self, mask: np.ndarray) -> "Detections":
        return Detections(self.boxes[mask], self.scores[mask], self.classes[mask])

    def to_list(self, class_names=CLASSES) -> list[dict]:
        return [{"class": class_names[int(c)], "class_id": int(c), "confidence": round(float(s), 4),
                 "box": [round(float(v), 1) for v in b]}
                for b, s, c in zip(self.boxes, self.scores, self.classes)]


# --------------------------------------------------------------------------------------------- #
# Image I/O and pre-processing
# --------------------------------------------------------------------------------------------- #
def read_image(path_or_bytes) -> np.ndarray:
    """BGR uint8, EXIF orientation ignored so pixels match the annotation coordinates."""
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), flags)
    else:   # np.fromfile + imdecode also handles non-ASCII Windows paths
        img = cv2.imdecode(np.fromfile(str(path_or_bytes), np.uint8), flags)
    if img is None:
        raise ValueError("could not decode image")
    return img


def to_blob(images_bgr: np.ndarray) -> np.ndarray:
    """[N, H, W, 3] BGR uint8 -> [N, 3, H, W] RGB float32 in 0..1 (contiguous)."""
    x = images_bgr[..., ::-1].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x, dtype=np.float32) * (1.0 / 255.0)


def letterbox(img: np.ndarray, size: int, color: int = 114) -> tuple[np.ndarray, float]:
    """Resize the long side to `size`, pad bottom/right to a square. Returns (image, scale)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), color, np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA)
    return out, r


def tile_origins(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """Top-left corners of a regular grid of `tile`-sized windows covering an h x w image."""
    stride = max(1, int(round(tile * (1.0 - overlap))))

    def axis(n):
        if n <= tile:
            return [0]
        pos = list(range(0, n - tile + 1, stride))
        if pos[-1] != n - tile:
            pos.append(n - tile)
        return pos

    return [(x, y) for y in axis(h) for x in axis(w)]


def make_tiles(img: np.ndarray, tile: int, overlap: float, pad_color: int = 114):
    """Returns tiles [N, tile, tile, 3] uint8 and their (x0, y0) origins [N, 2]."""
    h, w = img.shape[:2]
    if h < tile or w < tile:   # pad tiny images up to one tile
        padded = np.full((max(h, tile), max(w, tile), 3), pad_color, np.uint8)
        padded[:h, :w] = img
        img = padded
    H, W = img.shape[:2]
    origins = tile_origins(H, W, tile, overlap)
    tiles = np.stack([img[y:y + tile, x:x + tile] for x, y in origins])
    return tiles, np.asarray(origins, np.float32)


# --------------------------------------------------------------------------------------------- #
# Decoding and NMS
# --------------------------------------------------------------------------------------------- #
def nms(det: Detections, iou: float, max_det: int = 1000) -> Detections:
    """Class-aware NMS (OpenCV C++ implementation)."""
    if len(det) == 0:
        return det
    xywh = np.concatenate([det.boxes[:, :2], det.boxes[:, 2:] - det.boxes[:, :2]], 1)
    keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), det.scores.tolist(), det.classes.astype(np.int32).tolist(),
                                   0.0, iou)
    keep = np.asarray(keep, np.int64).reshape(-1)
    keep = keep[np.argsort(-det.scores[keep])][:max_det]
    return det.filter(keep)


def decode(output: np.ndarray, conf: float, iou: float = 0.6, max_det: int = 300,
           pre_nms_topk: int = 3000) -> list[Detections]:
    """Raw model output -> per-image Detections in model-input pixel coordinates."""
    output = np.asarray(output, dtype=np.float32)
    results = []
    if output.ndim == 3 and output.shape[-1] == 6 and output.shape[1] != 6:   # end-to-end head
        for o in output:
            o = o[o[:, 4] >= conf]
            results.append(Detections(o[:, :4].copy(), o[:, 4].copy(), o[:, 5].astype(np.int64)))
        return results
    pred = output.transpose(0, 2, 1)                       # [B, A, 4 + nc]
    for p in pred:
        cls_scores = p[:, 4:]
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls)), cls]
        m = score >= conf
        if not m.any():
            results.append(Detections.empty())
            continue
        p, cls, score = p[m], cls[m], score[m]
        if len(score) > pre_nms_topk:
            top = np.argpartition(-score, pre_nms_topk)[:pre_nms_topk]
            p, cls, score = p[top], cls[top], score[top]
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        results.append(nms(Detections(boxes, score, cls.astype(np.int64)), iou, max_det))
    return results


# --------------------------------------------------------------------------------------------- #
# Whole-board inference
# --------------------------------------------------------------------------------------------- #
def predict_full(run: RunFn, img: np.ndarray, size: int, conf: float, iou: float = 0.6,
                 max_det: int = 300) -> Detections:
    """Resize the whole board into one size x size input (the 'higher resolution' approach)."""
    lb, r = letterbox(img, size)
    det = decode(run(to_blob(lb[None])), conf, iou, max_det)[0]
    if len(det):
        det.boxes /= r
        h, w = img.shape[:2]
        det.boxes[:, [0, 2]] = det.boxes[:, [0, 2]].clip(0, w)
        det.boxes[:, [1, 3]] = det.boxes[:, [1, 3]].clip(0, h)
    return det


def predict_tiled(run: RunFn, img: np.ndarray, tile: int, overlap: float, conf: float,
                  iou: float = 0.6, merge_iou: float = 0.5, edge_margin: int = 3,
                  max_batch: int = 64, max_det: int = 1000, timings: dict | None = None) -> Detections:
    """Slice the board into overlapping native-resolution tiles, detect, and merge.

    Merging: a detection touching an *interior* tile edge is dropped, because the overlap
    guarantees that any defect smaller than the overlap is fully contained in a neighbouring
    tile. Duplicates from overlapping tiles are then removed by class-aware NMS.
    """
    import time
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    tiles, origins = make_tiles(img, tile, overlap)
    blob = to_blob(tiles)
    t1 = time.perf_counter()
    outs = [run(blob[i:i + max_batch]) for i in range(0, len(blob), max_batch)]
    out = np.concatenate(outs, 0)
    t2 = time.perf_counter()
    dets = decode(out, conf, iou, max_det=300)
    boxes, scores, classes = [], [], []
    for d, (x0, y0) in zip(dets, origins):
        if not len(d):
            continue
        b = d.boxes.copy()
        interior = np.zeros(len(b), bool)
        if x0 > 0:
            interior |= b[:, 0] <= edge_margin
        if y0 > 0:
            interior |= b[:, 1] <= edge_margin
        if x0 + tile < w:
            interior |= b[:, 2] >= tile - edge_margin
        if y0 + tile < h:
            interior |= b[:, 3] >= tile - edge_margin
        keep = ~interior
        b = b[keep] + np.array([x0, y0, x0, y0], np.float32)
        boxes.append(b); scores.append(d.scores[keep]); classes.append(d.classes[keep])
    if boxes:
        det = Detections(np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes))
        det.boxes[:, [0, 2]] = det.boxes[:, [0, 2]].clip(0, w)
        det.boxes[:, [1, 3]] = det.boxes[:, [1, 3]].clip(0, h)
        det = nms(det, merge_iou, max_det)
    else:
        det = Detections.empty()
    if timings is not None:
        t3 = time.perf_counter()
        timings["tiling_ms"] = (t1 - t0) * 1e3
        timings["inference_ms"] = (t2 - t1) * 1e3
        timings["postprocess_ms"] = (t3 - t2) * 1e3
        timings["num_tiles"] = len(tiles)
    return det


# --------------------------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------------------------- #
_PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72)]


def draw(img: np.ndarray, det: Detections, class_names=CLASSES, thickness: int | None = None) -> np.ndarray:
    out = img.copy()
    t = thickness or max(2, int(round(max(img.shape[:2]) / 600)))
    for b, s, c in zip(det.boxes.astype(int), det.scores, det.classes):
        col = _PALETTE[int(c) % len(_PALETTE)]
        cv2.rectangle(out, (b[0] - t, b[1] - t), (b[2] + t, b[3] + t), col, t)
        label = f"{class_names[int(c)]} {s:.2f}"
        fs = 0.4 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        y = max(b[1] - t - 4, th + 4)
        cv2.rectangle(out, (b[0] - t, y - th - 4), (b[0] - t + tw + 4, y + 2), col, -1)
        cv2.putText(out, label, (b[0] - t + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    max(1, t // 2), cv2.LINE_AA)
    return out
