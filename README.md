# FruArIC (Frugal Architecture for Identifying Container Codes)

Official implementation of the paper:

> **Matching model capacity to format constraints: a frugal architecture for container code identification on edge- oriented in port automation**

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
| CPU Latency | **13.955 ms** |

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
- Inference: Intel Ultra 7 265K CPU

---

# Experimental Results

<img width="692" height="449" alt="Image" src="https://github.com/user-attachments/assets/70fb6961-e38e-4006-9865-650469060d4a" />

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
python -m DataProcessing.DataProcessing_Experiments --src_img "benchmark_source/images" --src_json "benchmark_source/json" --exp_img "benchmark_selected/images" --exp_json "benchmark_selected/json" --processed_dir "datasets/benchmark/images"
```

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

#### 2.4 Text Recognizer Evaluation

```bash
python -m CharRec.test --FeatureExtraction CNN_s --output_channel 512 --eval_data "test_data_path" --saved_model "saved_models/CNN_s_fromDeformableCNN_s/best_accuracy.pth"
```

---

### 3. Benchmarking with `main.py`

Run the following command to benchmark the end-to-end pipeline combining the detector and text recognizer:

```bash
python -m main --image_folder "benchmark_data_path" --detector_path "detector_path" --recognizer_path "recognizer_path"
```

The filename stem of each benchmark image is used as its ground-truth string.

---
