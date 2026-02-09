# MRNet: Knee MRI Classification with PyTorch

PyTorch implementation of a deep learning model for detecting abnormalities in knee MRI scans using a ResNet-18 backbone with max pooling aggregation across slices.

## Overview

This project implements a binary classification model for knee MRI diagnosis across three tasks:
- **Abnormal detection**: General abnormality classification
- **ACL tear detection**: Anterior cruciate ligament tear identification
- **Meniscus tear detection**: Meniscal tear identification

The model processes 3D MRI volumes from three anatomical planes (sagittal, coronal, axial) using a 2D CNN with slice-level feature extraction and max pooling aggregation.

## Architecture

- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Aggregation**: Max pooling across MRI slices
- **Output**: Binary classification (abnormal/normal)

The model extracts features from each 2D slice independently, then aggregates using max pooling to capture the most salient abnormality indicators across the volume.

## Requirements

```
torch
torchvision
pandas
numpy
scikit-learn
```

## Dataset Structure

```
./
├── train/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
│       └── 0000.npy
├── valid/
│   ├── axial/
│   ├── coronal/
│   └── sagittal/
├── train-abnormal.csv
├── train-acl.csv
├── train-meniscus.csv
├── valid-abnormal.csv
├── valid-acl.csv
└── valid-meniscus.csv
```

Each `.npy` file contains a 3D MRI volume with shape `(slices, 256, 256)`. CSV files contain case IDs and binary labels.

## Usage

### Training

Configure the task and plane at the top of the script:

```python
TASK = 'abnormal'    # options: 'acl', 'meniscus', 'abnormal'
PLANE = 'sagittal'   # options: 'axial', 'coronal', 'sagittal'
```

Run training:

```bash
python mrnet.py
```

### Evaluation

After training, evaluate the model:

```python
test()
```

## Model Performance

The model outputs four metrics:
- **Loss**: Binary cross-entropy loss
- **Accuracy**: Classification accuracy
- **AUC**: Area under the ROC curve
- **F1 Score**: Harmonic mean of precision and recall

## Configuration

Key hyperparameters:

```python
LEARNING_RATE = 1e-5
EPOCHS = 1
BATCH_SIZE = 1  # fixed due to variable slice counts
```

## Implementation Details

- Input images are resized to 224×224 for ResNet-18 compatibility
- Grayscale MRI slices are replicated across 3 channels
- ImageNet normalization is applied for pretrained weights
- Adam optimizer with binary cross-entropy loss
- Batch size of 1 to handle variable slice depths per patient

## Model Checkpoints

Trained models are saved as `mrnet_{task}_{plane}.pth` (e.g., `mrnet_abnormal_sagittal.pth`).

## License

This implementation is for educational and research purposes.
