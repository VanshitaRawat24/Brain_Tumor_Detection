# Explainable Brain Tumor Detection Using Vision Transformers and Attention Mechanisms

## Overview

This project presents an explainable deep learning framework for **multiclass brain tumor classification from MRI images**, with a particular focus on understanding **where and why a model makes its predictions**.

The study investigates both convolutional and Transformer-based approaches for classifying three major brain tumor categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**

The project combines a strong CNN baseline with a custom **Vision Transformer (ViT)** and evaluates their explanation mechanisms using expert tumor segmentation masks from an external dataset.

The primary objective is not only to measure classification performance, but also to quantitatively evaluate whether model-generated explanations spatially correspond to regions identified as tumors by expert annotations.

---

## Research Objectives

The project addresses the following objectives:

1. Develop and evaluate deep learning models for multiclass brain tumor classification.
2. Establish a strong CNN-based baseline using transfer learning.
3. Implement a Vision Transformer with multi-head self-attention.
4. Analyze native Transformer attention for visual explanation.
5. Generate CNN explanations using Grad-CAM.
6. Compare CNN Grad-CAM and ViT attention spatially.
7. Validate explanation maps against expert tumor segmentation masks.
8. Perform an external, provenance-aware validation using the BRISC dataset.
9. Investigate the relationship between classification correctness and explanation-mask agreement.

---

## Datasets

### 1. Figshare Brain Tumor Dataset

The primary classification experiments use the publicly available Figshare brain tumor MRI dataset.

**Dataset source:**

https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

The dataset contains MRI images belonging to three tumor categories used in this study:

- Glioma
- Meningioma
- Pituitary

A total of **3,064 images** were processed for the experimental pipeline.

---

### 2. BRISC 2025 Dataset

The BRISC 2025 dataset is used as an external validation source for explanation analysis.

**Dataset paper:**

https://www.kaggle.com/datasets/briscdataset/brisc2025

BRISC provides expert-reviewed tumor segmentation masks along with MRI images. The segmentation subset contains **4,793 paired image-mask samples**, with **860 samples in the test split**.

The segmentation task includes:

- Glioma
- Meningioma
- Pituitary

The BRISC test set is used to compare model explanation maps against expert tumor masks.

---

## Dataset Overlap Audit

Before using BRISC for external validation, an explicit image-overlap audit was performed.

The audit included:

- Normalized SHA-256 comparison for exact image overlap.
- Perceptual hashing for near-duplicate candidate detection.
- Visual inspection of the strongest perceptual candidates.

Results:

- **0 exact normalized image overlaps** were detected.
- The strongest perceptual candidates were visually inspected.
- No visually confirmed duplicate was identified among the strongest candidates.

Therefore, BRISC was treated as a **provenance-aware external validation dataset** for the explanation analysis.

---

## Methodology

The overall experimental workflow consists of:

```text
MRI Dataset
     │
     ▼
Preprocessing
     │
     ├───────────────┐
     ▼               ▼
CNN Models         Vision Transformer
     │               │
     ▼               ▼
Grad-CAM          Multi-Head Self-Attention
     │               │
     └───────┬───────┘
             ▼
      Explanation Maps
             │
             ▼
   Expert Tumor Masks
             │
             ▼
 Spatial Agreement Analysis
```

---

## Image Preprocessing

MRI images are converted to a consistent input representation before model inference.

For the Vision Transformer:

```text
Original MRI
512 × 512
     ↓
Resize
128 × 128
     ↓
Grayscale channel
128 × 128 × 1
     ↓
Normalize
[0, 1]
```

The Vision Transformer uses a `128 × 128 × 1` input.

For ImageNet-pretrained CNN models such as VGG19, the grayscale image is replicated across three channels:

```text
128 × 128 × 1
        ↓
128 × 128 × 3
```

This allows the grayscale MRI data to be processed by architectures originally designed for RGB images.

---

## Deep Learning Models

Multiple CNN architectures and configurations were investigated, including:

- VGG16
- VGG19
- ResNet50
- ResNet101
- ResNet50 with SE attention
- ResNet50 with CBAM attention
- InceptionV3
- AlexNet
- LeNet

A Vision Transformer was additionally implemented to investigate Transformer-based image classification and native attention mechanisms.

---

## Vision Transformer

The custom Vision Transformer operates on `128 × 128` MRI images.

### Patch Embedding

The image is divided into non-overlapping `16 × 16` patches.

```text
128 / 16 = 8
```

Therefore:

```text
8 × 8 = 64 image patches
```

Each patch is projected into an embedding representation.

### Transformer Architecture

The implemented ViT includes:

- Patch embedding
- Learnable positional embeddings
- Multi-head self-attention
- Layer normalization
- Residual connections
- MLP blocks
- Dropout
- Classification head

The final ViT configuration uses a **CLS token**, resulting in:

```text
64 image patches + 1 CLS token = 65 tokens
```

With four attention heads, the attention representation can be analyzed across multiple attention heads and Transformer blocks.

---

## Explainability

Two explanation mechanisms are studied.

### CNN: Grad-CAM

For the CNN baseline, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is used to identify image regions that contribute to the predicted class.

The VGG19 explanation uses the final convolutional feature representation and produces a spatial heatmap that is resized to the original MRI resolution.

### Vision Transformer: Native Attention

For the Vision Transformer, attention scores are extracted directly from the model's **Multi-Head Self-Attention** layers.

The final Transformer block's CLS-to-patch attention is converted into an `8 × 8` spatial representation and resized to the MRI resolution.

This allows the spatial behavior of native Transformer attention to be investigated without replacing the attention mechanism with a separate post-hoc explanation method.

---

## Quantitative Explanation Evaluation

Instead of evaluating explanations only through visual inspection, the project compares explanation maps with expert tumor segmentation masks.

The following metrics are calculated:

### Dice Similarity Coefficient

Measures spatial overlap between the explanation region and the expert tumor mask.

\[
Dice = \frac{2|A \cap B|}{|A| + |B|}
\]

### Intersection over Union

Measures the intersection relative to the union of the two regions.

\[
IoU = \frac{|A \cap B|}{|A \cup B|}
\]

### Localization Recall

Measures how much of the expert tumor region is covered by the explanation.

\[
Recall = \frac{TP}{TP+FN}
\]

### Localization Precision

Measures how much of the highlighted explanation region corresponds to the expert tumor region.

\[
Precision = \frac{TP}{TP+FP}
\]

These metrics are interpreted as **spatial agreement/alignment with expert annotations**, rather than as direct measures of clinical faithfulness.

---

## Classification Evaluation

Models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix
- Per-class Precision
- Per-class Recall
- Per-class F1-score

Macro-averaged metrics are particularly important because the three tumor classes are not equally represented.

---

## CNN Baseline Performance

The strongest currently evaluated CNN configuration is VGG19 fine-tuning.

| Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1 (%) |
|---|---:|---:|---:|---:|
| VGG19 Fine-tuned | 91.50 | 87.98 | 90.01 | 88.79 |
| VGG16 Fine-tuned | 91.01 | 87.52 | 90.27 | 88.39 |
| ResNet50-SE Stage 1 | 90.85 | 87.30 | 90.32 | 88.41 |
| ResNet101 Stage 1 | 90.36 | 87.07 | 86.63 | 86.85 |
| ResNet50-CBAM Stage 1 | 88.89 | 85.32 | 87.90 | 86.33 |
| InceptionV3 Stage 1 | 86.76 | 82.71 | 86.11 | 83.61 |
| AlexNet | 81.37 | 78.71 | 83.67 | 79.49 |
| LeNet | 89.54 | 87.75 | 85.35 | 86.37 |

> Note: these results represent the current experimental comparison. Final paper reporting should follow the finalized validation-based model-selection protocol.

---

## Model Explainability Validation

The explanation validation compares:

```text
VGG19
   │
   └── Grad-CAM
          │
          ▼
     Explanation Map
          │
          ▼
     Expert Tumor Mask


ViT
   │
   └── Native CLS Attention
          │
          ▼
     Explanation Map
          │
          ▼
     Expert Tumor Mask
```

The two methods are evaluated on the same BRISC test samples using a matched thresholding protocol.

Paired statistical tests are used to determine whether observed differences in spatial agreement are systematic across the same images.

---

## Statistical Analysis

The explanation comparison uses paired statistical analysis because both explanation methods are evaluated on the same BRISC images.

The analysis includes:

- Paired Wilcoxon signed-rank tests
- Multiple-comparison correction using Holm's procedure
- Paired effect sizes
- Bootstrap confidence intervals

This provides statistical evidence in addition to reporting mean and median spatial agreement.

---

## Repository Structure

```text
Brain_Tumor_Detection/
│
├── data/
│   └── raw/
│       ├── Figshare_dataset/
│       └── brisc2025/
│
├── notebooks/
│   └── 03_deep_learning_models.ipynb
│
├── results/
│   ├── brisc_validation/
│   │   └── overlap_audit/
│   │
│   ├── model_comparison/
│   │
│   ├── transfer_models/
│   │
│   └── dataset_cache/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

Large datasets and generated artifacts may be excluded from the Git repository and obtained from their respective official sources.

---

## Reproducibility

The project uses fixed random seeds where applicable and stores intermediate experimental outputs for reproducibility.

Important experiment outputs include:

```text
figshare_dataset1_image_inventory.csv
figshare_dataset1_image_fingerprints.csv
dataset1_brisc_overlap_audit_summary.csv
brisc_provenance_external_validation_record.csv
brisc_segmentation_metadata.csv
brisc_gradcam_test_results.csv
brisc_vit_attention_test_results.csv
brisc_vit_attention_summary.csv
```

The notebook contains the implementation and evaluation workflow.

---

## Local Demonstration

A local demonstration interface is included to support image-level inference.

The intended workflow is:

```text
MRI Upload
    ↓
Preprocessing
    ↓
Trained Model
    ↓
Tumor Prediction
    ↓
Prediction Confidence
    ↓
Explanation Map
```

The interface is intended for research demonstration and **not for clinical diagnosis**.

---

## Limitations

The current study has several limitations:

- The datasets are research datasets rather than clinical deployment datasets.
- Dataset provenance does not guarantee complete subject-level independence across all sources.
- The custom ViT is trained from scratch and may underperform pretrained CNN architectures on the available dataset size.
- Attention maps should not automatically be interpreted as causal explanations.
- Spatial agreement with expert masks measures localization alignment, not clinical validity.
- Further multi-seed and cross-validation experiments are required for stronger generalization claims.
- Additional external datasets and prospective clinical evaluation would be required before any clinical application.

---

## Research Contribution

The central contribution of this project is the **quantitative study of model explanations alongside classification performance**.

Rather than treating explainability as only a visualization step, the framework evaluates:

1. CNN-based classification and Grad-CAM explanations.
2. Vision Transformer classification and native self-attention.
3. Spatial agreement between explanation maps and expert tumor masks.
4. Paired statistical comparison of the two explanation mechanisms.
5. External validation using a provenance-audited dataset.

This provides a structured framework for studying whether model-generated visual explanations correspond to independently annotated tumor regions.

---

## Disclaimer

This project is intended exclusively for **academic and research purposes**.

The models and explanations presented in this repository are not medical devices and should not be used for diagnosis, treatment decisions, or other clinical decisions.

---

## Authors

**Vanshita Rawat**  
**Harshita Bhardwaj**



---

## Acknowledgements

The project uses publicly available brain MRI datasets and builds upon established deep learning and explainability techniques including Vision Transformers, Multi-Head Self-Attention, Grad-CAM, and segmentation-based spatial evaluation.
