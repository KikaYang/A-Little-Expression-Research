# A-Little-Expression-Research

This project explores a full image-classification workflow for facial expression / emotion detection using:
* Hugging Face `datasets` (images stored as PIL objects inside dataset rows)
* PyTorch training loop written from scratch (`train_step`, `test_step`, `train`)
* A simple CNN baseline (TinyVGG) and a pretrained backbone (ResNet18)
* A custom `collate_fn` to bridge Hugging Face datasets with PyTorch `DataLoader`

The notebook documents an iterative debugging process: handling dataset format issues, fixing transforms, stabilizing evaluation, resolving model shape mismatches, and improving performance with better architectures.

---

## Dataset

The dataset is loaded from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("sxdave/emotion_detection")
```

Each sample has the structure:

```python
{"image": PIL.Image, "label": int}
```

The number of classes is extracted from the dataset:

```python
num_classes = ds["train"].features["label"].num_classes
```

---

## Key Design Choice: Custom `collate_fn`

Unlike `torchvision.datasets.ImageFolder`, Hugging Face datasets return dictionaries. A standard PyTorch `DataLoader` cannot automatically stack PIL images into tensors, so this project uses a **custom `collate_fn`** to:

1. apply transforms (PIL → Tensor)
2. stack tensors into a batch
3. create a label tensor

### Train / Eval transforms must be separated

A major pitfall is applying **random augmentations** to validation/test sets. The final setup uses:

* `train_transform` (may include random augmentation)
* `test_transform` (no random augmentation)

```python
def collate_fn(batch):
    images = [train_transform(x["image"]) for x in batch]
    labels = [x["label"] for x in batch]
    return torch.stack(images, 0), torch.tensor(labels)

def test_fn(batch):
    images = [test_transform(x["image"]) for x in batch]
    labels = [x["label"] for x in batch]
    return torch.stack(images, 0), torch.tensor(labels)
```

---

## DataLoaders

```python
from torch.utils.data import DataLoader

BATCH_SIZE = 16

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)

valid_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=test_fn)

test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, collate_fn=test_fn)
```

---

## Training Loop (From Scratch)

The project uses a classic structure:

* `train_step()` for forward + backward + update
* `test_step()` for evaluation with `torch.inference_mode()`
* `train()` to run epochs and record metrics

### `train_step()`

* `model.train()`
* forward pass
* `loss.backward()`
* `optimizer.step()`
* compute accuracy

### `test_step()`

* `model.eval()`
* `torch.inference_mode()`
* forward pass only
* compute loss + accuracy

### `train()`

Tracks results per epoch in a dict:

```python
results = {
  "train_loss": [], "train_acc": [],
  "test_loss": [], "test_acc": []
}
```

> Note: During training, the notebook passes `valid_dataloader` as the evaluation loader (named `test_dataloader` in the function signature). The actual test set should be evaluated only at the end.

---

## Baseline Model: TinyVGG

A lightweight CNN inspired by CNN Explainer’s TinyVGG, used as a baseline.

### Early issue: hardcoded classifier input size

A common CNN bug occurred when the classifier expected a fixed flattened size (depends on input resolution and padding). This was fixed by using **Adaptive Average Pooling**, so the classifier does not depend on spatial dimensions:

```python
self.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(hidden_units, output_shape)
)
```

### Improved TinyVGG (with BatchNorm)

BatchNorm was introduced for better training stability:

```python
nn.Conv2d(..., padding=1),
nn.BatchNorm2d(hidden_units),
nn.ReLU(),
```

---

## Pretrained Backbone: ResNet18

To improve generalization on a small dataset, the project switches to **ResNet18 with ImageNet pretrained weights**:

```python
from torchvision import models
model_1 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_1.fc = nn.Linear(model_1.fc.in_features, num_classes)
model_1 = model_1.to(device)
```

### Critical bug discovered: optimizer bound to the wrong model

At one point, training looked “stuck” because the optimizer was still created with `model_0.parameters()` (TinyVGG), so ResNet18 weights were not being updated.

Fix: always rebuild the optimizer when changing models:

```python
optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-3)
```

Or, for small datasets, train only the classification head:

```python
for p in model_1.parameters():
    p.requires_grad = False
for p in model_1.fc.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(model_1.fc.parameters(), lr=1e-3, weight_decay=1e-4)
```

---

## Observations & Lessons Learned

### 1) Small validation/test sets can cause unstable accuracy

When validation/test sets are small, accuracy changes in **large discrete steps** (a few images can move accuracy by several percent). This makes results appear “jumpy” even when training is stable.

### 2) Overfitting is easy on small datasets

Some runs reached **near-100% train accuracy** while validation accuracy stayed around ~0.5–0.6, indicating overfitting. This motivates:

* early stopping / best checkpoint saving
* weight decay
* stronger (but realistic) augmentation
* pretrained backbones

### 3) Always keep evaluation “clean”

Validation/test must not include random augmentation.

### 4) Always verify training is actually updating the right parameters

When results look suspiciously flat, check:

* `requires_grad` flags
* optimizer parameter groups
* gradient values on key layers

---

## How to Run

1. Install dependencies:

```bash
pip install torch torchvision datasets tqdm
```

2. Open and run the notebook:

* `Expression_detect.ipynb`

The notebook will:

* load the dataset from Hugging Face
* build dataloaders with custom `collate_fn`
* train TinyVGG baseline
* train ResNet18 pretrained model
* print epoch-by-epoch loss/accuracy

---

## Recommended Next Steps

* **Best-checkpoint saving** based on lowest validation loss
* **Fine-tuning**: unfreeze `layer4` of ResNet18 with a smaller learning rate (e.g. `1e-4`)
* **K-fold cross validation** for more reliable evaluation on small datasets
* Confusion matrix + per-class metrics (macro F1) to understand class-wise performance

---

## Project Structure

* `Expression_detect.ipynb` — main experiment notebook (data loading, training, debugging, results)
*  `README.md` — this file
