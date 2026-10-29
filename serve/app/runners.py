"""Inference backends for the PCB inspection API.

Each runner is a callable: float32 [B, 3, S, S] numpy blob -> raw model output (numpy).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("pcb")


class ORTRunner:
    def __init__(self, onnx_path: Path, device: str = "cuda"):
        import onnxruntime as ort
        if device == "cuda" and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                     else ["CPUExecutionProvider"])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
        active = self.sess.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"CUDAExecutionProvider not active (got {active})")
        self.input_name = self.sess.get_inputs()[0].name
        self.max_batch = 10**9
        self.description = f"onnxruntime-{device}"

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: blob})[0]


class TRTRunner:
    """TensorRT engine with CuPy device buffers allocated once at the maximum batch size."""

    def __init__(self, engine_path: Path):
        import cupy as cp
        import tensorrt as trt
        self.cp, self.trt = cp, trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.lock = threading.Lock()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if n != self.in_name)
        self.max_batch = int(self.engine.get_tensor_profile_shape(self.in_name, 0)[2][0])
        in_shape = (self.max_batch,) + tuple(self.engine.get_tensor_shape(self.in_name))[1:]
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        self.context.set_input_shape(self.in_name, in_shape)
        self.d_in = cp.empty(in_shape, dtype=self.in_dtype)
        self.d_out = cp.empty(tuple(self.context.get_tensor_shape(self.out_name)),
                              dtype=trt.nptype(self.engine.get_tensor_dtype(self.out_name)))
        self.description = f"tensorrt-{trt.__version__}"

    def __call__(self, blob: np.ndarray) -> np.ndarray:
        outs = []
        for i in range(0, len(blob), self.max_batch):       # chunk if a board has more tiles than max_batch
            outs.append(self._run(blob[i:i + self.max_batch]))
        return np.concatenate(outs, 0)

    def _run(self, blob: np.ndarray) -> np.ndarray:
        b = blob.shape[0]
        with self.lock, self.stream:
            self.d_in[:b].set(np.ascontiguousarray(blob, dtype=self.in_dtype), stream=self.stream)
            self.context.set_input_shape(self.in_name, blob.shape)
            self.context.set_tensor_address(self.in_name, self.d_in.data.ptr)
            self.context.set_tensor_address(self.out_name, self.d_out.data.ptr)
            if not self.context.execute_async_v3(self.stream.ptr):
                raise RuntimeError("TensorRT execution failed")
            out = self.d_out[:b].get(stream=self.stream)
            self.stream.synchronize()
        return out


def gpu_tag() -> str:
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    name = name.decode() if isinstance(name, bytes) else str(name)
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


def ensure_engine(onnx_path: Path, cache_dir: Path, max_batch: int, opt_batch: int) -> Path:
    """Build (once) and cache an engine keyed by model, TensorRT version and GPU."""
    import tensorrt as trt
    engine = cache_dir / f"{onnx_path.stem}_b{max_batch}_trt{trt.__version__}_{gpu_tag()}.engine"
    if engine.exists():
        return engine
    script = Path(__file__).with_name("build_trt_engine.py")
    log.info("Building TensorRT engine %s (first start only) ...", engine.name)
    cmd = [sys.executable, str(script), "--onnx", str(onnx_path), "--engine", str(engine),
           "--max-batch", str(max_batch), "--opt-batch", str(opt_batch)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    log.info(r.stdout.strip())
    if r.returncode != 0:
        raise RuntimeError(f"engine build failed ({r.returncode}):\n{r.stderr[-4000:]}")
    return engine
