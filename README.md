# A-Little-Expression-Research

This project explores a full image-classification workflow for facial expression / emotion detection using:
* Hugging Face `datasets` (images stored as PIL objects inside dataset rows)
* PyTorch training loop written from scratch (`train_step`, `test_step`, `train`)
* A simple CNN baseline (TinyVGG) and a pretrained backbone (ResNet18)
* A custom `collate_fn` to bridge Hugging Face datasets with PyTorch `DataLoader`

The notebook documents an iterative debugging process: handling dataset format issues, fixing transforms, stabilizing evaluation, resolving model shape mismatches, and improving performance with better architectures.
