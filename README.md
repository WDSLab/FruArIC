# FruArIC (Frugal Architecture for Identifying Container Codes)

Official implementation of the paper:

> **Matching Model Capacity to Format Constraints: A Frugal Architecture for Identifying Container Codes in Port Automation**

---

# Architecture

<img width="1412" height="1566" alt="Image" src="https://github.com/user-attachments/assets/7c61718b-2850-48a9-89f2-143acfe99298" />

FruArIC is a lightweight two-stage framework designed for container code recognition in resource-constrained edge environments.

- **TailoredDet**: P3-centric lightweight detector
- **PureConvSTR**: Convolution-only recognizer trained through hierarchical knowledge distillation
- **Format-Constrained Post-processing**

---

# Highlights

| Metric | FruArIC |
|----------|----------|
| Full-string Accuracy | **90.233%** |
| Parameters | **0.816 M** |
| MACs | **2.118 G** |
| CPU Latency | **67.030 ms** |

---

# Dataset

| Split | Images |
|---------|---------:|
| Train | 15,128 |
| Validation | 1,892 |
| Test | 1,891 |
| Independent Evaluation | 1,000 |

---


# Implementation Details

### TailoredDet

- Input size: 640×640 RGB
- Batch size: 16
- Learning rate: 0.01
- Optimizer: MuSGD
- Losses: CIoU, DFL, BCE

### PureConvSTR

- Input size: 150×45 Grayscale
- Batch size: 192
- Learning rate: 1.0
- Optimizer: Adadelta
- Recognition loss: CTC
- Distillation loss: KL Divergence

### Hardware

- Training: NVIDIA GeForce RTX 5080
- Inference: Intel Core Ultra 7 265K CPU (2 P-cores, 2 threads, maximum 3.7 GHz; 4 GiB combined committed-memory limit; FP32, batch size 1)

---

# Experimental Results

Results with format correction enabled, from Tables 4 and 5 of the manuscript. Latency is the mean end-to-end processing time, including preprocessing and postprocessing, on the constrained CPU configuration above.

| Detection | Recognition | Params (M) | Accuracy (%, mean (SD)) | MACs (G) | Latency (ms) |
|---|---|---:|---:|---:|---:|
| RT-DETR | ABINet | 69.547 | 83.533 (11.552) | 63.876 | 886.817 |
| RT-DETR | ASTER | 51.886 | 80.667 (6.224) | 57.426 | 813.305 |
| RT-DETR | SRN | 87.48 | 86.067 (7.955) | 58.446 | 819.647 |
| RT-DETR | SVTR | 38.845 | 51.222 (28.393) | 54.095 | 735.609 |
| YOLO26 | ABINet | 39.243 | 84.289 (8.396) | 12.766 | 232.658 |
| YOLO26 | ASTER | 21.582 | 81.444 (7.934) | 6.316 | 154.027 |
| YOLO26 | SRN | 57.176 | 84.933 (7.614) | 7.336 | 153.785 |
| YOLO26 | SVTR | 8.541 | 72.667 (6.811) | 2.985 | 97.173 |
| YOLOv12 | ABINet | 39.307 | 90.989 (1.810) | 13.120 | 257.652 |
| YOLOv12 | ASTER | 21.646 | 87.300 (2.107) | 6.670 | 176.075 |
| YOLOv12 | SRN | 57.24 | 92.133 (1.077) | 7.690 | 177.869 |
| YOLOv12 | SVTR | 8.605 | 85.211 (2.450) | 3.339 | 115.527 |
| YOLOv11 | ABINet | 39.329 | 91.733 (0.561) | 13.100 | 238.894 |
| YOLOv11 | ASTER | 21.667 | 87.744 (1.180) | 6.650 | 159.050 |
| YOLOv11 | SRN | 57.262 | 92.044 (1.226) | 7.670 | 164.120 |
| YOLOv11 | SVTR | 8.627 | 83.933 (2.214) | 3.320 | 101.762 |
| YOLOv8-SL | SVTR-R | 8.520 | 83.256 (3.865) | 3.840 | 133.116 |
| TailoredDet | PureConvSTR | 0.816 | 90.233 (1.989) | 2.118 | 67.030 |

---


# Usage

All commands must be executed from the repository root directory

### 1. Data Preprocessing

#### 1.1 Source Data Structure

The source data must consist of JPG images and JSON annotation files with matching filenames.

```text
source_data/
├── images/
│   ├── sample_0001.jpg
│   ├── sample_0002.jpg
│   └── ...
└── json/
    ├── sample_0001.json
    ├── sample_0002.json
    └── ...
```

For example, an image and a JSON file with the same filename stem are treated as an image–annotation pair.

#### 1.2 Source Data Cleaning

```bash
python -m DataProcessing.DataProcessing_Src --src_img "source_data/images" --src_json "source_data/json"
```

#### 1.3 Detector Dataset Preparation

```bash
python -m DataProcessing.DataProcessing_RegionDet --src_img "source_data/images" --src_json "source_data/json" --dir_data "datasets/region_detection"
```

The dataset is organized as follows:

```text
datasets/region_detection/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

#### 1.4 Text Recognition Dataset Preparation

```bash
python -m DataProcessing.DataProcessing_CharRec --src_img "source_data/images" --src_json "source_data/json" --work_dir "datasets/character_recognition"
```

The dataset is organized as follows:

```text
datasets/character_recognition/
├── srcs/
├── train/
├── validation/
├── test/
├── gt/
│   ├── gt_train.txt
│   ├── gt_validation.txt
│   └── gt_test.txt
└── lmdb/
    ├── train/
    ├── validation/
    └── test/
```

#### 1.5 Benchmark Dataset Preparation

```bash
python -m DataProcessing.DataProcessing_Experiments --src_img "benchmark_source/images" --src_json "benchmark_source/json" --exp_img "benchmark_selected/images" --exp_json "benchmark_selected/json" --processed_dir "datasets/benchmark/images" --mode random --seed 1111
```

The seed above is an example, not a documented seed for the paper's evaluation set. Specify `--mode random --seed <seed>` for seeded random sampling or `--mode uniform` for evenly spaced selection; the mode is required.

Each benchmark image filename must contain its ground-truth container identification code.

```text
datasets/benchmark/images/
├── ABCU1234567.jpg
├── BCIU7654321.jpg
└── ...
```

---

### 2. Model Training

#### 2.1 Code Region Detector Training

```bash
python -m RegionDet.train --model_cfg "RegionDet/cfg/TailoredDet.yaml" --data "data.yaml" --epochs 100 --patience 10 --imgsz 640 --name TailoredDet
```

#### 2.2 Text Recognizer Training

The text recognizers are trained in the following order:

```text
DeformableGRCNN_l
        ↓
 DeformableCNN_l
        ↓
 DeformableCNN_s
        ↓
      CNN_s
```

##### Step 1: Train DeformableGRCNN-l

```bash
python -m CharRec.train --FeatureExtraction DeformableGRCNN_l --output_channel 1280 --train_data "train_data_path" --valid_data "valid_data_path" 
```

##### Step 2: DeformableGRCNN-l → DeformableCNN-l

```bash
python -m CharRec.train --FeatureExtraction DeformableCNN_l --output_channel 1280 --train_data "train_data_path" --valid_data "valid_data_path" --distillation --teacher_path "saved_models/DeformableGRCNN_l/best_accuracy.pth" --teacher_backbone DeformableGRCNN_l --teacher_output 1280 --copy_weights
```

##### Step 3: DeformableCNN-l → DeformableCNN-s

```bash
python -m CharRec.train --FeatureExtraction DeformableCNN_s --output_channel 512 --train_data "train_data_path" --valid_data "valid_data_path" --distillation --teacher_path "saved_models/DeformableCNN_l_fromDeformableGRCNN_l/best_accuracy.pth" --teacher_backbone DeformableCNN_l --teacher_output 1280
```

##### Step 4: DeformableCNN-s → CNN-s

```bash
python -m CharRec.train --FeatureExtraction CNN_s --output_channel 512 --train_data "train_data_path" --valid_data "valid_data_path" --distillation --teacher_path "saved_models/DeformableCNN_s_fromDeformableCNN_l/best_accuracy.pth" --teacher_backbone DeformableCNN_s --teacher_output 512 --copy_weights
```

---

### 3. Benchmarking with `main.py`

Run the following command to benchmark the end-to-end pipeline combining the detector and text recognizer:

```bash
python -m main --image_folder "benchmark_data_path" --detector_path "detector_path" --recognizer_path "recognizer_path"
```

The filename stem of each benchmark image is used as its ground-truth string.

---
