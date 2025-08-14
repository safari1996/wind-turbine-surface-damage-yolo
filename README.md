# wind-turbine-surface-damage-yolo
End-to-end YOLO pipeline for detecting and evaluating wind-turbine surface/blade damage (training, validation, inference, and visualization).
# YOLO Wind Turbine Surface Damage (WTSD)

End-to-end pipeline for **annotating, training, validating, and deploying** a YOLO-based model to detect **wind-turbine surface/blade damage** from UAV or telephoto imagery. Includes dataset layout, commands, evaluation scripts, and result tables aligned with the *3.5.paper script* experiment logs (e.g., `yolo_m_150e`, `val_std`, `test_std`).

> **TL;DR**: Clone, prepare `wtsd.yaml`, drop your images/labels into `datasets/wtsd`, and run the training command below to reproduce the baseline `yolov8m` (150 epochs) results.

---

## 1) Features

* ✅ YOLOv8 training/validation/inference commands (Ultralytics)
* ✅ Clean dataset structure with example `wtsd.yaml`
* ✅ Reproducible experiment setup: **`yolo_m_150e`**
* ✅ Evaluation helpers for **Precision / Recall / mAP50 / mAP50–95**
* ✅ Prediction & overlay visualization (images + videos)
* ✅ Export to ONNX / TensorRT for edge deployment

---

## 2) Repository structure (suggested)

```
.
├── datasets/
│   └── wtsd/
│       ├── images/
│       │   ├── train/  # *.jpg / *.png
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/  # *.txt in YOLO format (one box per line)
│           ├── val/
│           └── test/
├── configs/
│   └── wtsd.yaml       # dataset YAML (paths + class names)
├── experiments/
│   ├── yolo_m_150e/    # training run folder (results.csv, weights, plots)
│   ├── val_std/        # validation logs/exports used in paper
│   └── test_std/       # test logs/exports used in paper
├── weights/
│   └── yolo_m_150e.pt  # (optional) trained checkpoint(s)
├── src/
│   ├── train_yolo_wtsd.py
│   ├── evaluate_yolo_wtsd.py
│   ├── infer_yolo_wtsd.py
│   └── visualize_wtsd_predictions.py
├── README.md
└── LICENSE
```

---

## 3) Dataset

### 3.1 YAML (`configs/wtsd.yaml`)

```yaml
# Example dataset config
path: ./datasets/wtsd
train: images/train
val: images/val
# optional test split (used for final reporting)
test: images/test

names:
  0: surface_damage    # ← replace with your exact class list if multi-class
  # 1: crack
  # 2: erosion
  # 3: discoloration
  # ...
```

### 3.2 Annotation format (YOLO)

Each line in a label file: `class x_center y_center width height` (normalized in \[0,1]). Example:

```
0 0.512 0.433 0.081 0.049
```

> **Tip**: Keep **val** images representative of lighting, angles, and turbine types to avoid evaluation bias.

---

## 4) Quickstart

### 4.1 Requirements

* Python ≥ 3.9
* PyTorch (CUDA optional but recommended)
* Ultralytics (YOLOv8)

```bash
# create env (recommended)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate

pip install --upgrade pip
pip install ultralytics==8.*
```

### 4.2 Verify install

```bash
python -c "from ultralytics import YOLO; print(YOLO('yolov8m.pt'))"
```

---

## 5) Train

Baseline experiment: **`yolo_m_150e`** (YOLOv8m, 150 epochs, 640px).

```bash
# From repo root
ultralytics detect train \
  model=yolov8m.pt \
  data=configs/wtsd.yaml \
  imgsz=640 \
  epochs=150 \
  batch=16 \
  device=0 \
  project=experiments \
  name=yolo_m_150e
```

> Adjust `batch` to fit your GPU memory. For CPU-only, remove `device=0` (training will be slower).

---

## 6) Validate & Test

Validate on **val** split:

```bash
ultralytics detect val \
  model=experiments/yolo_m_150e/weights/best.pt \
  data=configs/wtsd.yaml \
  imgsz=640 \
  split=val \
  project=experiments \
  name=val_std
```

Evaluate on **test** split for final reporting:

```bash
ultralytics detect val \
  model=experiments/yolo_m_150e/weights/best.pt \
  data=configs/wtsd.yaml \
  imgsz=640 \
  split=test \
  project=experiments \
  name=test_std
```

Artifacts of interest (Ultralytics-standard):

* `experiments/<run>/results.csv` – per-epoch metrics
* `experiments/<run>/weights/best.pt` – best checkpoint
* `experiments/<run>/confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, etc.

---

## 7) Inference

Single image or folder:

```bash
ultralytics detect predict \
  model=experiments/yolo_m_150e/weights/best.pt \
  source=sample_images/ \
  imgsz=640 \
  conf=0.25 \
  project=experiments \
  name=predictions
```

Video/stream (e.g., drone feed):

```bash
ultralytics detect predict \
  model=experiments/yolo_m_150e/weights/best.pt \
  source=sample_videos/windfarm.mp4 \
  stream=False
```

> Outputs (with bounding boxes + confidences) are saved under `experiments/predictions/`.

---

## 8) Visualization utilities

```bash
# Export annotated example grids
python src/visualize_wtsd_predictions.py \
  --images experiments/predictions \
  --out runs/vis_grid.png
```

```bash
# Plot PR / F1 curves from a given run
python src/evaluate_yolo_wtsd.py \
  --results_csv experiments/yolo_m_150e/results.csv \
  --out_dir experiments/yolo_m_150e/plots
```

---

## 9) Results 

> Replace the placeholders below with your logged values from `val_std` and `test_std` (e.g., taken from Ultralytics summaries or your exported tables). Keep the exact metric definitions and splits for consistency with the paper.

### 9.1 Validation (val\_std)

| Split | Model   | img size | Epochs |   mAP\@50 | mAP\@50–95 | Precision |    Recall |
| ----: | ------- | -------: | -----: | --------: | ---------: | --------: | --------: |
|   val | yolov8m |      640 |    150 | `<0.678>` |  `<0.468>` | `<0.615>` | `<0.722>` |

### 9.2 Test (test\_std)

| Split | Model   | img size | Epochs |   mAP\@50 | mAP\@50–95 | Precision |    Recall |
| ----: | ------- | -------: | -----: | --------: | ---------: | --------: | --------: |
|  test | yolov8m |      640 |    150 | `<0.705>` |  `<0.497>` | `<0.692>` | `<0.787>` |

### 9.3 Confusion matrix (test)

Include or regenerate from Ultralytics output:

```
experiments/test_std/confusion_matrix.png
```

### 9.4 Qualitative examples

Place a few representative before/after frames (UAV nadir/oblique, different lighting) under:

```
assets/qualitative/
```

Reference them in your paper or GitHub README as needed.

---

## 10) Reproducing the `yolo_m_150e` run exactly

* **Seed**: set `seed=42` in CLI or `cfg.yaml` (for strict reproducibility).
* **Augmentations**: Use Ultralytics defaults unless your paper specifies otherwise.
* **Hyperparameters**: If you tuned LR, momentum, or schedulers, commit the `.yaml` to `experiments/yolo_m_150e/hyp.yaml` and reference it here.

Example with explicit seed + cache:

```bash
ultralytics detect train \
  model=yolov8m.pt \
  data=configs/wtsd.yaml \
  imgsz=640 \
  epochs=150 \
  batch=16 \
  seed=42 \
  cache=True \
  project=experiments \
  name=yolo_m_150e
```

---

## 11) Export for deployment

```bash
# ONNX (dynamic batch/shape optional)
ultralytics export model=experiments/yolo_m_150e/weights/best.pt format=onnx opset=12

# TensorRT (for Jetson / edge)
ultralytics export model=experiments/yolo_m_150e/weights/best.pt format=engine
```

> Consider INT8/FP16 quantization for edge devices. Validate accuracy drop on **test** before production.

---

## 12) Known limitations & roadmap

* Small, hairline cracks at long range may require **higher input resolution** or **tiling**.
* Domain shifts (blade color, camera sensor, glare) can reduce recall; consider **domain-specific fine-tuning**.
* Multi-class damage taxonomy (crack/erosion/discoloration/LEP) planned; update `names:` in `wtsd.yaml` accordingly.

**Planned**:

* Multi-class labels & per-class AP table
* Mosaic/Copy-Paste ablation
* Semi-supervised fine-tuning with pseudo-labels

---

## 13) Citation

If you use this repository in academic work, please cite:

```bibtex
@software{WTSD_YOLO_2025,
  title        = {YOLO Wind Turbine Surface Damage (WTSD)},
  author       = {MohammadMahdiSafari},
  year         = {2025},
  url          = {[https://github.com/<your-username>/<your-repo>](https://github.com/safari1996/wind-turbine-surface-damage-yolo)},
  note         = {Version: yolo\_m\_150e baseline}
}
```

---

## 14) License

Choose a license and place it in `LICENSE` (e.g., MIT/Apache-2.0). Note dataset licenses may differ.

---

## 15) Acknowledgements

* Ultralytics YOLOv8
* UAV pilots & annotators who prepared the **WTSD** dataset

---

### Appendix A — Handy snippets

**Count class distribution:**

```python
# tools/count_classes.py
from pathlib import Path
from collections import Counter

root = Path('datasets/wtsd/labels/train')
ctr = Counter()
for p in root.glob('*.txt'):
    for line in p.read_text().strip().splitlines():
        cls = int(line.split()[0])
        ctr[cls] += 1
print(ctr)
```

**Merge val/test metrics into a Markdown table:**

```python
# tools/metrics_to_md.py
import pandas as pd

val = pd.read_csv('experiments/val_std/results.csv').iloc[[-1]]
val['Split'] = 'val'

tst = pd.read_csv('experiments/test_std/results.csv').iloc[[-1]]
tst['Split'] = 'test'

cols = ['Split', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']
df = pd.concat([val, tst])[cols]
df.columns = ['Split','mAP@50','mAP@50-95','Precision','Recall']
print(df.to_markdown(index=False))
```

