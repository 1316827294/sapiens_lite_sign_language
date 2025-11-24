---

# 🔧 Generate Keypoints with Sapiens Lite

This guide explains how to generate human keypoints using the **Sapiens Lite** model with GPU-accelerated multi-process video processing.

---

## 📦 1. Environment Setup

Create and activate the conda environment:

```bash
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
```

Install PyTorch with CUDA 12.1 support:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install additional dependencies:

```bash
pip install opencv-python tqdm json-tricks
```

---

## 📁 2. Navigate to the Lite Directory

Go to the directory containing the **lite** toolkit:

```bash
git clone https://github.com/facebookresearch/sapiens.git
cd path/to/sapiens/lite
```

---

## 🛠️ 3. Configure `multiprocess_video_processor.py`

Open the file and update the following paths to match your environment:

```python
POSE_CHECKPOINT = "/home/yunli/data/sapiens/lite/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"
DATASET_ROOT    = "/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game/"
OUTPUT_ROOT     = "/home/yunli/data/sapiens/lite/game/"
```

Make sure:

* `POSE_CHECKPOINT` points to the correct **TorchScript pose model (.pt2)**
* `DATASET_ROOT` contains your input videos
* `OUTPUT_ROOT` is where JSON keypoint results will be saved

---

## ▶️ 4. Run the Pipeline

Execute the processing script:

```bash
bash game.sh
```

This will launch the multi-process keypoint extraction system using **8 CPU cores + GPU** for optimal speed.

---

## 📌 Results

* For each input video, a corresponding JSON result file will be generated under:

```
$OUTPUT_ROOT/<train|val|test>/<category>/<video>.json
```

* Each JSON contains:

  * video metadata
  * per-frame keypoints
  * confidence scores

---
