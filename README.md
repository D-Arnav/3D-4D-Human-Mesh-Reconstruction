# 3D and 4D Human Mesh Reconstruction using SMPL / SMPLify

This project provides a compact implementation of SMPL-based human body modeling and single-image mesh reconstruction. The code follows the core ideas of the SMPL model and the SMPLify optimization framework, focusing on readability and reproducibility rather than speed. It supports mesh rendering, pose/shape manipulation, camera estimation, and SMPL parameter optimization from 2D joints.

---

## Features

* Load and manipulate SMPL meshes (shape β and pose θ).
* Render T-pose meshes or arbitrary joint configurations.
* Fit SMPL parameters to 2D keypoints (projection + priors).
* Basic camera translation/orientation estimation.
* Visualizations of predicted joints, loss curves, and reconstructed meshes.

---

## Model Downloads

The SMPL model used is licensed separately by the Max Planck Institute for Intelligent Systems for research and educational use only. The model files are **not** included in this repository. Please register and download them from the official websites:

* [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
* [https://smplify.is.tue.mpg.de/index.html](https://smplify.is.tue.mpg.de/index.html)

Once downloaded, place the following files in the `data/` directory:

```
data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
data/gmm_08.pkl
```

---

## Installation

```bash
conda create --name human python=3.10 -y
conda activate human

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

pip install -e .[all]
```

To verify the environment:

```bash
python render_smpl_tpose.py
```

A rendered neutral T-pose mesh should appear in `results/smpl_tpose.png`.

---

## Basic Usage

### **Render a custom shape or pose**

```bash
python render_smpl_tpose.py --beta 1.5 0 0 0 0 0 0 0 0 0
python render_pose.py --pose_file example_pose.npy
```

### **Run SMPLify-style single-image fitting**

(Uses LSP 2D joints or OpenPose joints, depending on directories provided.)

```bash
python fit_smplify.py --viz
```

This produces:

* Camera-estimation visualizations
* Reprojection overlays
* Final SMPL mesh renderings

Outputs are saved under:

```
results/lsp/smplify/
results/lsp/smplify_reproj_only/
```

### **Render only (no optimization)**

```bash
python fit_smplify.py --viz --render_only
```

---

## Video Frame Reconstruction

The pipeline also supports running on selected frames from a video:

```bash
python fit_smplify.py --viz \
  --img_dir example_data/videos/example_1/images \
  --joints_dir example_data/videos/example_1/joints \
  --dataset openpose \
  --selected_indices 0,5,10,18
```

Results are saved under:

```
results/example_1/smplify/
```
