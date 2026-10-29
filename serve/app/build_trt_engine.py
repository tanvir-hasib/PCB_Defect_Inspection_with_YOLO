"""Build a TensorRT engine from an ONNX model in a clean, isolated process.

Precision comes from the ONNX graph itself (TensorRT strong typing):
  * model_fp32.onnx  -> fp32 engine
  * model_fp16.onnx  -> fp16 engine (ModelOpt AutoCast graph)
  * model_int8.onnx  -> INT8 engine (explicit Q/DQ nodes from ModelOpt PTQ; no calibrator needed at build time)
Works with TensorRT 10 (STRONGLY_TYPED flag) and TensorRT 11 (strong typing is the only mode).

Running it as a separate process keeps PyTorch / ONNX Runtime CUDA DLLs out of TensorRT's process
(a common cause of hard crashes on Windows), and a native crash cannot kill a Jupyter kernel.

Usage:
    python tools/build_trt_engine.py --onnx model_fp16.onnx --engine model_fp16.engine \
        --max-batch 32 --opt-batch 18 [--gpu 0] [--verbose]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--opt-batch", type=int, default=1)
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    ap.add_argument("--gpu", default=None, help="nvidia-smi index; default: inherit CUDA_VISIBLE_DEVICES")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[build] python {sys.version.split()[0]} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    import tensorrt as trt
    print(f"[build] TensorRT {trt.__version__}", flush=True)
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if int(trt.__version__.split(".")[0]) < 11:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    print(f"[build] parsing {args.onnx}", flush=True)
    if not parser.parse_from_file(str(args.onnx)):
        for i in range(parser.num_errors):
            print("[build] parser error:", parser.get_error(i), file=sys.stderr)
        return 2

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    inp = network.get_input(0)
    rest = tuple(inp.shape)[1:]
    if any(d < 0 for d in rest):
        print(f"[build] only the batch axis may be dynamic, got {tuple(inp.shape)}", file=sys.stderr)
        return 2
    opt = max(1, min(args.opt_batch, args.max_batch))
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, *rest), (opt, *rest), (args.max_batch, *rest))
    config.add_optimization_profile(profile)

    print(f"[build] building {inp.name} {tuple(inp.shape)} batch 1..{args.max_batch} (opt {opt}) ...", flush=True)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[build] build_serialized_network returned None (see TensorRT messages above)", file=sys.stderr)
        return 3
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    dt = time.perf_counter() - t0

    engine = trt.Runtime(logger).deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        print("[build] engine written but cannot be deserialised", file=sys.stderr)
        return 4
    io = [f"{engine.get_tensor_name(i)}:{engine.get_tensor_dtype(engine.get_tensor_name(i))}"
          for i in range(engine.num_io_tensors)]
    print(f"[build] OK in {dt:.0f}s -> {args.engine} ({args.engine.stat().st_size / 2**20:.1f} MB) | io: {io}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
