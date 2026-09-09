# Hotel Review Rating Classification
### Frozen BERT Embeddings, Classical ML and DistilBERT Fine-Tuning

Comparative study of neural and classical approaches for **five-class hotel review rating prediction**, using pretrained Transformer representations and task-specific fine-tuning.

The project evaluates three strategies:

1. **Frozen BERT embeddings + MLP**
2. **Frozen BERT embeddings + classical scikit-learn classifiers**
3. **End-to-end DistilBERT fine-tuning**

It also includes an HPC-oriented execution pipeline with embedding caching, checkpointing, automatic resume and Slurm signal handling.

> **Academic context:** Master 1 Artificial Intelligence, Avignon Université, 2025-2026  
> **Course:** Neural Approaches / Machine Learning  
> **Frameworks:** PyTorch, Hugging Face Transformers, scikit-learn  
> **Author:** Steve Sanogo

---

## Overview

The objective is to predict the rating associated with an English hotel review:

```text
1 star
2 stars
3 stars
4 stars
5 stars
```

The task is formulated as a **five-class text classification problem**.

Rather than evaluating a single architecture, the notebook studies how performance changes when moving from:

```text
Frozen general-purpose representations
                |
                v
Simple downstream classifiers
                |
                v
Task-specific Transformer fine-tuning
```

The final comparison shows that adapting the Transformer itself provides the strongest improvement.

---

# Dataset

The dataset is divided into three predefined splits:

| Split | Reviews |
|---|---:|
| Train | **18,491** |
| Validation | **1,000** |
| Test | **1,000** |

Each sample contains:

```text
Review : hotel review text
Rating : integer from 1 to 5
```

Ratings are converted to class indices:

```text
1 -> 0
2 -> 1
3 -> 2
4 -> 3
5 -> 4
```

---

## Class Imbalance

The dataset is noticeably imbalanced.

In the training set:

```text
1 star : 1,310
2 stars: 1,630
3 stars: 1,982
4 stars: 5,465
5 stars: 8,104
```

The 5-star class represents approximately **44% of the training data**.

On the test set, predicting only the majority class would already yield approximately:

```text
46.6% accuracy
```

This makes per-class precision, recall and F1 important in addition to overall accuracy.

---

## Review Length

The average review length is:

| Split | Mean words |
|---|---:|
| Train | **102.3** |
| Validation | **135.8** |
| Test | **111.0** |

The training set contains reviews as long as approximately 1,900 words.

---

# Approach 1: Frozen BERT Embeddings + MLP

## BERT Representation

The project uses:

```text
bert-base-uncased
```

as a frozen feature extractor.

The tokenizer converts raw text into token IDs and attention masks.

BERT then produces contextual representations of dimension:

```text
768
```

for each token.

The notebook applies **mean pooling over non-padding tokens** to obtain a single 768-dimensional representation per review.

```text
Review
  |
Tokenizer
  |
bert-base-uncased
  |
Token embeddings
  |
Mean pooling
  |
768-dimensional review embedding
```

BERT weights remain frozen during this stage.

---

## Embedding Cache

Computing BERT embeddings is one of the most expensive steps.

To avoid recomputing them for every experiment, embeddings are cached to disk:

```text
Train embeddings : 18,491 x 768
Valid embeddings : 1,000 x 768
Test embeddings  : 1,000 x 768
```

If cached tensors already exist, they are loaded directly.

This separation makes subsequent experiments with MLP and classical classifiers significantly faster.

---

## MLP Architecture

The baseline MLP receives the 768-dimensional BERT embedding and predicts five logits.

```text
768-dimensional BERT embedding
             |
             v
        Linear 256
             |
            ReLU
             |
         Dropout
             |
        Linear 128
             |
            ReLU
             |
         Dropout
             |
         Linear 5
             |
             v
      Rating prediction
```

The baseline configuration uses:

```text
hidden layers : 256 -> 128
dropout       : 0.3
loss          : CrossEntropyLoss
optimizer     : Adam
```

---

## Baseline Result

The baseline MLP reaches:

| Metric | Test result |
|---|---:|
| Accuracy | **61.40%** |
| Macro F1 | **0.51** |

Per-class behavior shows that the model performs best on the dominant 5-star class:

```text
5-star F1: 0.79
```

while intermediate ratings remain more difficult, particularly the 3-star class.

---

# MLP Hyperparameter Search

A two-phase search strategy is used.

## Phase 1: Rapid Screening

The active grid evaluates:

```text
hidden_dim : 256, 512
dropout    : 0.1, 0.2, 0.3
lr         : 1e-3, 5e-3, 1e-2
batch_size : 32
```

giving:

```text
18 configurations
```

Each configuration is trained for at most five epochs with early stopping.

---

## Phase 2: Final Training

The best screening configuration is:

```text
hidden_dim : 512
dropout    : 0.2
lr         : 0.001
batch_size : 32
```

It reaches:

```text
Best validation accuracy : 63.20%
Test accuracy            : 60.80%
Macro F1                 : 0.51
```

The grid search does not improve over the original MLP baseline.

This suggests that the main performance limitation lies in the **frozen representation**, rather than in the downstream MLP architecture.

---

# Approach 2: Classical ML on BERT Embeddings

The same frozen 768-dimensional BERT embeddings are converted to NumPy arrays and evaluated with multiple scikit-learn classifiers.

The project compares:

- Logistic Regression
- RBF SVC
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors

---

## Validation Comparison

| Classifier | Validation accuracy |
|---|---:|
| Logistic Regression | **61.7%** |
| SVC, RBF | **60.7%** |
| Gradient Boosting | **58.7%** |
| Random Forest | **56.1%** |
| KNN, k=7 | **51.6%** |

Logistic Regression and SVC are selected for further tuning.

---

## GridSearchCV

### Logistic Regression

Search:

```text
C = 0.01, 0.1, 1.0, 10.0
```

Best configuration:

```text
C = 0.1
```

Cross-validation accuracy:

```text
60.75%
```

### SVC

Search:

```text
C     = 0.1, 1.0, 10.0
gamma = scale, auto
```

Best configuration:

```text
C     = 10.0
gamma = auto
```

Cross-validation accuracy:

```text
60.69%
```

---

## Best Classical Model

The selected Logistic Regression model reaches:

```text
Test accuracy: 61.10%
```

This is very close to the frozen-BERT MLP baseline.

---

# Approach 3: DistilBERT Fine-Tuning

The third approach removes the frozen-representation constraint.

The notebook fine-tunes:

```text
distilbert-base-uncased
```

directly on the five-class rating task.

A classification head is added using:

```python
AutoModelForSequenceClassification(
    num_labels=5
)
```

---

## Fine-Tuning Configuration

```text
max sequence length : 256
batch size          : 16
epochs              : 4
learning rate       : 2e-5
weight decay        : 0.01
optimizer           : AdamW
scheduler           : linear decay with warm-up
```

The model contains approximately:

```text
66.96 million trainable parameters
```

---

## Validation Progress

| Epoch | Train accuracy | Validation accuracy |
|---:|---:|---:|
| 1 | 58.70% | 66.00% |
| 2 | 69.28% | 68.40% |
| 3 | 75.82% | **68.70%** |
| 4 | 81.16% | 67.40% |

The best validation model occurs at epoch 3.

---

## Final DistilBERT Result

| Metric | Test result |
|---|---:|
| Accuracy | **68.50%** |
| Macro F1 | **0.62** |
| Weighted F1 | **0.68** |

Selected per-class F1 scores:

```text
1 star : 0.62
2 stars: 0.56
3 stars: 0.55
4 stars: 0.57
5 stars: 0.81
```

The largest improvement appears in the intermediate rating classes.

For example:

```text
3-star F1
Frozen BERT + MLP : 0.21
Fine-tuned DistilBERT: 0.55
```

---

# Final Comparison

| Approach | Test Accuracy | Macro F1 |
|---|---:|---:|
| Frozen BERT + MLP baseline | **61.4%** | 0.51 |
| Frozen BERT + tuned MLP | 60.8% | 0.51 |
| Frozen BERT + Logistic Regression | 61.1% | not reported |
| **Fine-tuned DistilBERT** | **68.5%** | **0.62** |

The fine-tuned Transformer improves absolute test accuracy by approximately:

```text
+7 percentage points
```

over the strongest frozen-embedding approaches.

---

# Why Fine-Tuning Helps

The frozen approaches all share the same representation bottleneck.

```text
Generic BERT embeddings
        |
        +--> MLP
        |
        +--> Logistic Regression
        |
        +--> SVC
```

Changing the classifier only produces limited gains because the representation itself remains fixed.

Fine-tuning changes the internal Transformer representations:

```text
Hotel reviews
      |
      v
DistilBERT
weights updated for the rating task
      |
      v
Task-specific representations
      |
      v
5-class classifier
```

This allows the model to better distinguish subtle semantic differences between ratings such as 2, 3 and 4 stars.

---

# HPC and Slurm Engineering

A major part of this project concerns execution robustness in constrained compute environments.

The notebook includes support for:

- GPU execution
- cached embeddings
- checkpoint persistence
- automatic checkpoint resume
- Slurm signal interception
- clean interruption of long-running jobs
- recovery of optimizer state
- recovery of scheduler state for Transformer fine-tuning

---

## Slurm Signal Handling

Before terminating a scheduled GPU job, Slurm can send:

```text
SIGUSR1
```

The notebook installs a signal handler that marks the training loop for graceful shutdown.

The current state is then saved before termination.

---

## Automatic Resume

At startup, the training pipeline checks whether a checkpoint already exists.

If present, it restores:

```text
model parameters
optimizer state
scheduler state
current epoch
best validation score
best model state
```

Training can therefore resume without restarting from epoch 0.

This is implemented for both:

- the MLP pipeline
- DistilBERT fine-tuning

---

# Reproducibility

The project fixes random seeds for:

- Python
- NumPy
- PyTorch
- CUDA

The recorded environment includes:

```text
PyTorch : 2.10.0+cu128
GPU     : Tesla T4
VRAM    : approximately 15.6 GB
```

The notebook automatically uses CUDA when available.

---

# Recommended Repository Structure

```text
hotel-review-rating-classification/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── hotel_review_rating_classification.ipynb
│
├── figures/
│   ├── class_distribution.png
│   ├── confusion_matrix_distilbert.png
│   └── model_comparison.png
│
└── data/
    └── README.md
```

If the original dataset cannot be redistributed, the `data/README.md` file should explain the expected filenames and local directory structure.

---

# Installation

Main dependencies:

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib
```

For reproducibility, use exact versions in a `requirements.txt`.

---

# Running the Notebook

The notebook expects three CSV files:

```text
train_hotel_reviews.csv
valid_hotel_reviews.csv
test_hotel_reviews.csv
```

Each must contain:

```text
Review
Rating
```

The pipeline can then be executed in the following order:

```text
1. Environment and reproducibility setup
2. Data loading and exploration
3. Frozen BERT embedding extraction
4. MLP baseline
5. MLP hyperparameter search
6. Classical classifier comparison
7. DistilBERT fine-tuning
8. Final test comparison
```

---

# Computational Trade-Off

The project highlights an important deployment trade-off.

| Criterion | Frozen BERT Embeddings | Fine-Tuned DistilBERT |
|---|---|---|
| Test accuracy | about 61% | **68.5%** |
| Macro F1 | 0.51 | **0.62** |
| Training cost | Low after embedding extraction | Higher |
| GPU requirement | Optional for downstream classifiers | Recommended |
| Representation | Fixed | Task-adapted |
| Reusability of embeddings | High | Lower |
| Intermediate rating discrimination | Limited | Better |

Frozen embeddings are attractive when computational cost and reuse are priorities.

Fine-tuning is preferable when predictive quality is more important.

---

# What This Project Demonstrates

This project demonstrates the ability to:

- formulate sentiment analysis as five-class rating prediction
- analyze class imbalance
- work with pretrained Transformer models
- extract and cache BERT embeddings
- implement MLP classifiers in PyTorch
- benchmark neural and classical ML models
- conduct hyperparameter search
- use GridSearchCV
- fine-tune DistilBERT end to end
- analyze per-class precision, recall and F1
- build confusion matrices
- manage GPU workloads
- implement checkpointing and automatic resume
- design Slurm-resilient training workflows
- compare model performance against computational cost

---

# Key Result

> **Fine-tuned DistilBERT achieved 68.5% test accuracy and 0.62 macro F1 on five-class hotel review rating prediction, improving accuracy by approximately 7 percentage points over frozen BERT embedding approaches.**

---

# Academic Context

**Master 1, Artificial Intelligence**  
**Avignon Université, CERI**  
**2025-2026**

This project was completed as part of coursework on neural approaches to supervised learning.

**Author:** Steve Sanogo
