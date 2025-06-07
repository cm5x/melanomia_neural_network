# Melanoma Classification with Pretrained AlexNet

## Overview

This project implements a melanoma vs. benign skin lesion classification model using transfer learning with a pretrained AlexNet network. We employ stratified k-fold cross-validation on the training set and evaluate on a held-out test set. Key features:

* **Transfer Learning**: Fine-tune the final fully-connected layer of ImageNet-pretrained AlexNet.
* **Stratified K-Fold**: Preserve class balance across folds.
* **Data Augmentation**: Random flips, rotations, color jitter.
* **Metrics & Visualization**: Track training/validation loss and accuracy per fold, plot average curves, and compute confusion matrix, precision, recall, and F1-score on test data.

## Dataset

Images are organized into two folders (`train` and `test`), each containing subfolders for the two classes:

```
train/
├── benign/
│   ├── img1.jpg
│   └── ...
└── malignant/
    ├── img99.jpg
    └── ...

test/
├── benign/
└── malignant/
```

### Data Sources

* [ISIC 2020 Challenge (33,126 images)](https://challenge2020.isic-archive.com/)
* [Kaggle Melanoma Skin Cancer Dataset (10,000 images)](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/)
* [Mendeley Melanoma Dataset](https://data.mendeley.com/datasets/ggh6g39ps2/3)

> **Note**: Download and arrange images into `train/` and `test/` folders as shown above.

## Requirements

* Python 3.8+
* GPU-enabled environment (e.g., Colab)
* PyTorch >=1.7
* torchvision
* scikit-learn
* matplotlib
* seaborn
* tqdm

Install dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn tqdm torchsummary
```

## Usage

1. **Mount Drive** (Colab):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Set path variables** in the notebook:

   ```python
   train_folder_path = '/content/drive/MyDrive/.../train'
   test_folder_path  = '/content/drive/MyDrive/.../test'
   ```
3. **Configure hyperparameters** (batch size, learning rate, epochs, k-folds).
4. **Run notebook** cells sequentially:

   * Data preprocessing
   * Stratified k-fold cross-validation with pretrained AlexNet
   * Model training & validation
   * Cross-validation results visualization
   * Final test evaluation & metrics

The script will output per-epoch metrics for each fold, followed by average curves and final test accuracy, confusion matrix, and classification metrics.

## File Structure

```
├── README.md               # This document
├── melanoma_training.ipynb # Main Colab notebook
└── model.pth               # Saved trained model weights
```

## Results

Example final test accuracy: **95.3%**
Example F1-score: **0.94**

*Adjust hyperparameters and augmentation strategies to improve performance further.*

## License

MIT License
