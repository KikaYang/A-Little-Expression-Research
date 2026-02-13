# A-Little-Expression-Research

This project documents the full engineering process of building an image classification system for facial expression detection using PyTorch.

Rather than starting from a pretrained model, the workflow begins with a manually implemented CNN (TinyVGG-style architecture) and gradually evolves toward a transfer learning solution (ResNet18).

The repository captures:

* Dataset handling using Hugging Face `datasets`
* Custom PyTorch training loop (no high-level trainer)
* Debugging model shape mismatches
* Handling small-dataset instability
* Transition from scratch CNN → pretrained backbone
* Building a minimal inference app for demonstration

This is not just a demo app — it is a structured exploration of deep learning system design.

---

# Project Philosophy

Instead of directly importing a pretrained model and calling `.fit()`, this project was built step-by-step:

1. Implement a baseline CNN from scratch
2. Write custom `train_step` / `test_step` loops
3. Understand batching and transforms deeply
4. Identify overfitting and data-size limitations
5. Introduce pretrained transfer learning
6. Wrap the final model into a lightweight demo interface

Each stage reveals a specific engineering or modeling insight.

---

# Stage 1: Custom TinyVGG (From Scratch)

The first implementation was a small CNN inspired by TinyVGG:

* Two convolutional blocks
* ReLU activations
* MaxPooling
* Fully connected classifier

Early issues encountered:

* Shape mismatch when flattening
* Hard-coded linear input size
* Sensitivity to input resolution

This was solved by introducing:

```python
nn.AdaptiveAvgPool2d((1,1))
```

which removes dependency on spatial input size.

BatchNorm was later added to improve stability.

---

# Stage 2: Custom Training Pipeline

Instead of using high-level training wrappers, the project defines:

* `train_step()`
* `test_step()`
* `train()`

Key learning points:

* Correct device placement
* Separating `model.train()` and `model.eval()`
* Using `torch.inference_mode()` for evaluation
* Manual accuracy calculation
* Tracking metrics per epoch

---

# Stage 3: Hugging Face Dataset Integration

Unlike `ImageFolder`, Hugging Face datasets return dictionary samples:

```python
{"image": PIL.Image, "label": int}
```

To integrate with PyTorch `DataLoader`, a custom `collate_fn` was required to:

* Apply transforms
* Stack tensors
* Construct label tensors

Critical realization:

Validation and test sets must NOT use random augmentation.

---

# Stage 4: Overfitting & Dataset Size Effects

On the small dataset:

* Training accuracy quickly reached ~100%
* Validation accuracy fluctuated significantly
* Small test sets caused discrete accuracy jumps

This revealed:

* The danger of overfitting
* The instability of evaluation on small datasets
* The importance of best-checkpoint saving

---

# Stage 5: Transition to Transfer Learning

To improve generalization, the model was switched to:

```python
torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
```

Key lessons:

* Always rebuild the optimizer when switching models
* Freeze backbone first, then fine-tune
* Use official ImageNet normalization for pretrained models
* Transfer learning significantly stabilizes validation accuracy

This marked the shift from experimental CNN design to practical deep learning engineering.

---

# Final Output: Minimal Inference App

The final trained model is wrapped in a simple Gradio app.

The goal is demonstration — not production deployment.

The app:

* Loads trained checkpoint
* Applies correct preprocessing
* Returns predicted label and class probabilities

---

# How To Run

There are two ways to explore this project:

---

## Option 1 — Explore the Full Development Process

Open the notebook:

```bash
jupyter notebook Expression_detect.ipynb
```

The notebook walks through:

* Dataset loading
* Transform design
* TinyVGG implementation
* Debugging issues
* ResNet18 transition
* Training logs and evaluation

This is recommended if you want to understand the full reasoning process.

---

## Option 2 — Run the Demo App

Install dependencies:

```bash
pip install torch torchvision datasets gradio tqdm pillow
```

Run:

```bash
python demo.py
```

Open in browser:

```
http://127.0.0.1:7860
```

Upload an image to see prediction results.

---

# Technical Stack

* Python 3.x
* PyTorch
* Torchvision
* Hugging Face Datasets
* Gradio

---

# Key Engineering Takeaways

* Writing your own training loop builds deeper understanding.
* Small datasets amplify evaluation instability.
* Adaptive pooling prevents classifier shape bugs.
* Pretrained models dramatically improve small-data performance.
* Always verify optimizer parameter bindings.
* Separate train and evaluation transforms strictly.
