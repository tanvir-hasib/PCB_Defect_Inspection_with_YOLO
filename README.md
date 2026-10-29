# PCB Defect Inspection with YOLO
### Tiled small-defect detection · AOI metrics (escape rate / false calls) · TensorRT INT8 · FastAPI + Docker

Automated optical inspection (AOI) of bare printed circuit boards: a YOLO detector finds six defect types and reports them the way an inspection-machine vendor would, by **escape rate**, **false calls per board** and **inspection time per board**, not only mAP. The model is quantised to **INT8 with explicit calibration** for TensorRT and served behind a FastAPI service in Docker.

> **Placeholders:** values shown as `—` are filled in from `artifacts/results/` after running the notebook.

| Highlight | Value |
|---|---|
| Deployed variant (chosen on validation) | `—` |
| Test mAP50 / mAP50-95 | `—` / `—` |
| Escape rate at ≤ 0.2 false calls per board (test) | `—` % |
| Inspection time per board, TensorRT INT8 (p50) | `—` ms (`—` boards/hour) |
| INT8 vs fp32: change in escape rate / mAP50 | `—` / `—` |

---

## Contents
- [Problem](#problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Engineering notes](#engineering-notes)
- [Limitations and future work](#limitations-and-future-work)
- [References and licences](#references-and-licences)

---

## Problem

The six defect classes are `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur` and `spurious_copper`. Original PKU-Market-PCB boards are about 3000 × 2000 px, while a typical defect is 20–60 px wide, **well under 1 % of the board width**. Resizing a whole board to a standard detector input shrinks most defects to a few pixels. The project compares three ways of handling this with the same model:

| Variant | Training input | Inference |
|---|---|---|
| `full640` | whole board resized to 640 px (baseline) | whole board at 640 |
| `full1280` | whole board resized to 1280 px | whole board at 1280 |
| `tiled640` | 640 × 640 crops at native resolution | overlapping 640 px tiles (25 % overlap), merged back to board coordinates |

---

## Dataset

**PKU-Market-PCB** (Huang & Wei, 2019; also known as HRIPCB): 693 annotated board images from 10 board designs, with synthetically added defects, about 115 images per class.

| | Value |
|---|---|
| Source used here | `—` (original release / Roboflow export) |
| Image resolution used | `—` |
| Images (train / val / test) | `—` / `—` / `—` |
| Defects | `—` |
| Median defect size | `—` px (`—` % of board width) |
| Defect pixels as share of board area | `—` % |

**Split:** stratified by defect class (70 / 15 / 15), and made **by original board**. The Roboflow export placed augmented copies of the same board in train, validation and test (308 boards leaked across splits). The notebook detects this and re-splits so that all copies of a board stay in one split. Validation and test use one image per board.

**Class imbalance:** the six defect classes are roughly balanced. The imbalance that matters is defect vs background, which is what drives false calls. In tiled training it is controlled explicitly by the ratio of background tiles to defect tiles.

---

## Approach

1. **Training:** Ultralytics `yolo11s`, identical hyper-parameters for all three variants, horizontal and vertical flips added to the defaults.
2. **Board-level evaluation:** every variant is scored the same way (whole board in, defects in board coordinates out) through one shared module, [`serve/app/pcb_core.py`](serve/app/pcb_core.py). The API uses that same module, so offline metrics and production behaviour come from identical code.
3. **Tile merging:** detections touching an *interior* tile edge are dropped, because the 25 % overlap guarantees each defect also appears whole in a neighbouring tile. Duplicates are removed with class-aware NMS.
4. **AOI metrics:**
   - **Escape rate:** the share of real defects that no detection overlaps. Matching is class-agnostic with IoU ≥ 0.25: a defect detected with the wrong class still stops the board.
   - **False calls per board:** detections overlapping no real defect.
   - **Operating point:** the lowest confidence threshold with ≤ 0.2 false calls per board on the **validation** set. It is then frozen and applied unchanged to the test set.
5. **Export:**
   - ONNX fp32 (fixed spatial size, dynamic batch).
   - **fp16** via NVIDIA ModelOpt AutoCast, which keeps overflow-prone layers in fp32.
   - **INT8** via ModelOpt post-training quantization: entropy calibration on training tiles writes explicit Q/DQ nodes into the graph, and the Detect head is kept in fp16.
6. **TensorRT:** strongly typed engines built from each ONNX file (fp32, fp16, INT8), in an isolated process.

---

## Results

All numbers come from a single run of [`notebooks/pcb_defect_yolo.ipynb`](notebooks/pcb_defect_yolo.ipynb). Raw values are in [`results/`](results/).

**Test environment:** NVIDIA GeForce RTX 2080 Ti (Turing, 11 GB), driver 617.42, Windows, Python 3.10, PyTorch `—`, Ultralytics `—`, TensorRT 11.4.0.106, ModelOpt `—`.

### 1. Tiling vs higher resolution (test set, PyTorch fp32)

| Variant | mAP50 | mAP50-95 | AP small | Escape rate @ op. point | False calls / board | Board time p50 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| full640 | — | — | — | — % | — | — |
| full1280 | — | — | — | — % | — | — |
| tiled640 | — | — | — | — % | — | — |

![Escape rate vs false calls, per-class AP](results/figures/variant_comparison.png)

### 2. Per-class AP50 (test)

| Class | full640 | full1280 | tiled640 |
|---|---:|---:|---:|
| missing_hole | — | — | — |
| mouse_bite | — | — | — |
| open_circuit | — | — | — |
| short | — | — | — |
| spur | — | — | — |
| spurious_copper | — | — | — |

### 3. Deployment backends (deployed variant, test boards)

| Backend | mAP50 | Escape @ op. point | False calls / board | Board time p50 (ms) | p95 (ms) | Boards / hour |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch fp32 | — | — % | — | — | — | — |
| PyTorch fp16 | — | — % | — | — | — | — |
| ONNX Runtime CUDA fp32 | — | — % | — | — | — | — |
| TensorRT fp32 | — | — % | — | — | — | — |
| TensorRT fp16 | — | — % | — | — | — | — |
| **TensorRT INT8** | — | — % | — | — | — | — |

Board time covers the whole inspection: image decoding, tiling, inference on all tiles, and merging.

![Backend benchmark](results/figures/backend_benchmark.png)

### 4. API load test (FastAPI, TensorRT INT8)

| Concurrent clients | Boards / s | Boards / hour | p50 (ms) | p95 (ms) |
|---:|---:|---:|---:|---:|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |

### 5. Error analysis

Missed defects and highest-confidence false calls at the operating point:

![Escapes](results/figures/escapes.png)
![False calls](results/figures/false_calls.png)

---

## Repository structure

```
pcb-aoi/
├── notebooks/
│   └── pcb_defect_yolo.ipynb    # full pipeline with a step-by-step guide
├── serve/
│   ├── app/
│   │   ├── pcb_core.py          # tiling, decoding, NMS, merging: shared by notebook and API
│   │   ├── runners.py           # TensorRT (CuPy) and ONNX Runtime backends, engine cache
│   │   ├── build_trt_engine.py  # builds strongly typed TensorRT engines in an isolated process
│   │   └── main.py              # FastAPI endpoints
│   ├── Dockerfile               # CUDA 13 runtime, non-root user, health check
│   ├── docker-compose.yml       # GPU selection, engine-cache volume
│   └── requirements.txt
├── results/                     # metrics CSVs and figures from the run above
├── requirements.txt             # notebook / training environment
└── LICENSE
```

Generated at run time and not committed: `artifacts/` (datasets, training runs, ONNX models, TensorRT engines).

---

## Quick start

### Requirements
- NVIDIA GPU with compute capability ≥ 7.5 (developed on an RTX 2080 Ti)
- **NVIDIA driver R580 or newer** (required by the CUDA 13 TensorRT wheel)
- Python 3.10–3.12
- PKU-Market-PCB, downloaded separately (not included in this repository)

### 1. Install

```bash
git clone https://github.com/<your-username>/pcb-aoi.git
cd pcb-aoi
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
# Linux:   export PCB_ROOT=/path/to/dataset
# Windows: set PCB_ROOT=C:\path\to\dataset
jupyter lab notebooks/pcb_defect_yolo.ipynb
```

`PCB_ROOT` can be either the original release (`images/` + `Annotations/`) or a Roboflow YOLO export (`data.yaml`); the format is detected automatically. Set `GPU_ID` in the first code cell on multi-GPU machines. Training all three variants takes about 2.5–3 h on an RTX 2080 Ti and can be resumed.

### 3. Serve the API

**Without Docker** (Windows cmd; on Linux use `export`):

```bat
pip install fastapi "uvicorn[standard]" python-multipart cupy-cuda13x
cd serve
set MODEL_DIR=..\artifacts\deploy\pcb
set ENGINE_CACHE=..\artifacts\engines\serve
set BACKEND=trt
set PRECISION=int8
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0
uvicorn app.main:app --port 8000
```

**With Docker** (NVIDIA Container Toolkit on Linux; Docker Desktop with WSL2 on Windows):

```bash
cd serve
docker compose build
GPU_ID=0 PRECISION=int8 docker compose up -d
docker compose logs -f api        # wait for "Ready"
```

The first start builds the TensorRT engine for the GPU it runs on (1–5 min); restarts reuse the cached engine. Interactive docs: <http://127.0.0.1:8000/docs>

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND` | `trt` | `trt`, `ort-cuda` or `ort-cpu` |
| `PRECISION` | `fp16` | `int8`, `fp16` or `fp32` |
| `CONF` | from `meta.json` | override the validated operating threshold |
| `GPU_ID` | `0` | (compose only) GPU index as listed by `nvidia-smi` |

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, backend, runtime |
| GET | `/model` | Classes, tiling settings, operating point, test metrics |
| POST | `/inspect` | Board image → `PASS`/`FAIL` with a defect list and timings; `?return_image=true` adds an annotated JPEG; `?conf=` overrides the threshold |
| POST | `/inspect/annotated` | Board image → annotated JPEG; verdict in the `X-Verdict` header |

```bash
curl -F "file=@board.jpg" http://127.0.0.1:8000/inspect
```

```json
{
  "verdict": "FAIL",
  "num_defects": 2,
  "defects": [
    {"class": "spur", "class_id": 4, "confidence": 0.91, "box": [1532.0, 845.5, 1561.2, 873.0]},
    {"class": "short", "class_id": 3, "confidence": 0.87, "box": [2210.4, 1120.0, 2244.8, 1166.3]}
  ],
  "conf_threshold": 0.31,
  "image_size": [3034, 1586],
  "timing_ms": {"decode_ms": 0.0, "tiling_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "num_tiles": 0, "total_ms": 0.0}
}
```

The response above illustrates the format only; the values are not measured results.

---

## Engineering notes

- **One inference core for evaluation and production.** Tiling, decoding, NMS and merging live in `pcb_core.py` and are imported by both the notebook and the API, so the reported metrics describe exactly what the service does.
- **INT8 in TensorRT 11 needs explicit Q/DQ.** The builder-side INT8 calibrator was removed in TensorRT 11. ModelOpt post-training quantization calibrates on representative tiles and writes Q/DQ nodes into the ONNX graph, and TensorRT builds the INT8 engine from that graph without any calibrator.
- **Accuracy is checked at the operating point, not only by mAP.** INT8 is accepted or rejected on its escape rate at the frozen threshold, because a small mAP change can hide extra escapes on the smallest defect class.
- **Leakage control.** Roboflow augmentations (`<name>.rf.<hash>`) are grouped by original board before splitting.
- **Engines are not portable.** They are tied to the GPU and the TensorRT version, so the container ships ONNX files and builds and caches engines on first start.
- **Isolated engine builds.** Engines are built in a separate process, so PyTorch's CUDA libraries never share a process with TensorRT's, and a native crash cannot kill the Jupyter kernel or the API.

---

## Limitations and future work

- **Dataset.** PKU-Market-PCB has synthetically added defects on only 10 board designs, and the same designs appear in train and test (as in per-product AOI recipes). Generalisation to new designs and to real production defects is not measured.
- **Few good boards.** False calls are measured on the defect-free areas of defective boards. A production system would be validated on thousands of good boards.
- **Lenient matching.** Escapes use IoU ≥ 0.25 because the boxes are tiny. Stricter localisation requirements would lower the numbers.
- **Next steps:**
  - GPU JPEG decoding (nvJPEG), since CPU decoding of large boards limits throughput;
  - dynamic batching across boards;
  - quantization-aware training if INT8 costs too much accuracy on the smallest defects;
  - reference-image comparison using golden-board templates.

---

## References and licences

- W. Huang, P. Wei. *A PCB Dataset for Defects Detection and Classification.* arXiv:1901.08204, 2019.
- R. Ding, L. Dai, G. Li, H. Liu. *TDD-Net: A Tiny Defect Detection Network for Printed Circuit Boards.* CAAI Transactions on Intelligence Technology, 2019.
- F. C. Akyon, S. O. Altinuc, A. Temizel. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (SAHI).* ICIP 2022.

**Code:** AGPL-3.0, required because the project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is licensed under AGPL-3.0. A commercial product would need an Ultralytics Enterprise licence or a permissively licensed detector (e.g. YOLOX, RT-DETR).
**Dataset:** PKU-Market-PCB is not included in this repository; obtain it from its original source and follow its terms of use.

**Author:** Tanvir Hasib, MSc (Information Technology) candidate, Multimedia University, Malaysia
