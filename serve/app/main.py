"""FastAPI service: PCB defect inspection (YOLO, tiled inference, TensorRT / ONNX Runtime).

Environment variables
---------------------
MODEL_DIR     folder with meta.json and model_{fp32,fp16,int8}.onnx            [/models]
ENGINE_CACHE  where TensorRT engines are built and cached                      [/cache]
BACKEND       trt | ort-cuda | ort-cpu                                         [trt]
PRECISION     int8 | fp16 | fp32                                               [fp16]
CONF          decision threshold override (default: operating point in meta)  [meta]
MAX_UPLOAD_MB reject larger uploads                                            [30]
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from .pcb_core import Detections, draw, predict_full, predict_tiled, read_image
from .runners import ORTRunner, TRTRunner, ensure_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pcb")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
ENGINE_CACHE = Path(os.getenv("ENGINE_CACHE", "/cache"))
BACKEND = os.getenv("BACKEND", "trt")
PRECISION = os.getenv("PRECISION", "fp16")
MAX_UPLOAD = int(float(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024)

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    onnx_path = MODEL_DIR / f"model_{PRECISION}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    if BACKEND == "trt":
        ENGINE_CACHE.mkdir(parents=True, exist_ok=True)
        engine = ensure_engine(onnx_path, ENGINE_CACHE, int(meta["max_tiles"]), int(meta["typical_tiles"]))
        runner = TRTRunner(engine)
    elif BACKEND in ("ort", "ort-cuda"):
        runner = ORTRunner(onnx_path, "cuda")
    elif BACKEND == "ort-cpu":
        runner = ORTRunner(onnx_path, "cpu")
    else:
        raise ValueError(f"unknown BACKEND={BACKEND!r}")
    conf = float(os.getenv("CONF", meta["operating_point"]["conf_threshold"]))
    # warm-up with a board-sized dummy image (autotuning, lazy allocation)
    dummy = np.full((meta["typical_board_hw"][0], meta["typical_board_hw"][1], 3), 114, np.uint8)
    STATE.update(meta=meta, runner=runner, conf=conf)
    for _ in range(2):
        _inspect_array(dummy, conf)
    log.info("Ready: %s, %s, %s, conf=%.3f", meta["model_name"], runner.description, PRECISION, conf)
    yield
    STATE.clear()


app = FastAPI(title="PCB Defect Inspection API", version="1.0.0", lifespan=lifespan)


def _inspect_array(img: np.ndarray, conf: float) -> tuple[Detections, dict]:
    m, run = STATE["meta"], STATE["runner"]
    timings: dict = {}
    if m["mode"] == "tiled":
        det = predict_tiled(run, img, m["tile"], m["overlap"], conf, iou=m["nms_iou"],
                            merge_iou=m["merge_iou"], edge_margin=m["edge_margin"],
                            max_batch=int(m["max_tiles"]), timings=timings)
    else:
        t0 = time.perf_counter()
        det = predict_full(run, img, m["imgsz"], conf, iou=m["nms_iou"])
        timings["inference_ms"] = (time.perf_counter() - t0) * 1e3
    return det, timings


def _inspect_bytes(data: bytes, conf: float):
    t0 = time.perf_counter()
    try:
        img = read_image(data)
    except ValueError as e:
        raise HTTPException(400, "Could not decode image") from e
    t1 = time.perf_counter()
    det, timings = _inspect_array(img, conf)
    timings = {"decode_ms": (t1 - t0) * 1e3, **timings}
    return img, det, timings


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"Upload larger than {MAX_UPLOAD // (1024 * 1024)} MB")
    return data


@app.get("/health")
def health():
    m = STATE.get("meta", {})
    return {"status": "ok" if STATE else "loading", "backend": BACKEND, "precision": PRECISION,
            "runtime": STATE["runner"].description if STATE else None, "model": m.get("model_name")}


@app.get("/model")
def model_info():
    m = STATE["meta"]
    return {k: m[k] for k in ("model_name", "classes", "mode", "tile", "overlap", "imgsz", "operating_point",
                              "test_metrics") if k in m} | {"conf_threshold_in_use": STATE["conf"]}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...),
                  conf: float | None = Query(None, ge=0.0, le=1.0, description="override decision threshold"),
                  return_image: bool = Query(False, description="include an annotated JPEG (base64)")):
    t_start = time.perf_counter()
    data = await _read(file)
    thr = STATE["conf"] if conf is None else conf
    img, det, timings = await run_in_threadpool(_inspect_bytes, data, thr)
    resp = {
        "verdict": "FAIL" if len(det) else "PASS",
        "num_defects": len(det),
        "defects": det.to_list(STATE["meta"]["classes"]),
        "conf_threshold": thr,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
    }
    if return_image:
        jpg = await run_in_threadpool(_annotated_jpeg, img, det)
        resp["annotated_jpeg_base64"] = base64.b64encode(jpg).decode()
    timings["total_ms"] = (time.perf_counter() - t_start) * 1e3
    resp["timing_ms"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in timings.items()}
    return resp


@app.post("/inspect/annotated", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
async def inspect_annotated(file: UploadFile = File(...), conf: float | None = Query(None, ge=0.0, le=1.0)):
    data = await _read(file)
    thr = STATE["conf"] if conf is None else conf
    img, det, _ = await run_in_threadpool(_inspect_bytes, data, thr)
    jpg = await run_in_threadpool(_annotated_jpeg, img, det)
    return Response(jpg, media_type="image/jpeg",
                    headers={"X-Verdict": "FAIL" if len(det) else "PASS", "X-Num-Defects": str(len(det))})


def _annotated_jpeg(img: np.ndarray, det: Detections) -> bytes:
    ok, buf = cv2.imencode(".jpg", draw(img, det, STATE["meta"]["classes"]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "JPEG encoding failed")
    return buf.tobytes()
