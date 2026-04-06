# Code References & Comments Documentation

This document contains all comments, docstrings, and implementation notes extracted from the codebase for reference purposes.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Module-by-Module Documentation](#module-by-module-documentation)
3. [Data Pipeline & Distortions](#data-pipeline--distortions)
4. [Training Architecture](#training-architecture)
5. [Network Components](#network-components)
6. [Configuration & Options](#configuration--options)
7. [Constants & Magic Numbers](#constants--magic-numbers)

---

## Project Overview

### DDP Training for Contrastive Learning
- **File**: `main_contrast.py`
- **Purpose**: Distributed Data Parallel (DDP) training for contrastive learning
- **Key Features**:
  - Multi-GPU training support via DDP
  - Distributed sampling for proper batch distribution
  - Automatic gradient synchronization across nodes

### Re-IQA (Image Quality Assessment) with Contrastive Learning
The project trains image encoders using contrastive learning specifically designed for image quality assessment. Both quality-aware and content-aware approaches are supported.

---

## Module-by-Module Documentation

### Demo Scripts

#### `demo_quality_aware_feats.py`
- **Purpose**: Extract quality-aware features from images
- **Key Comments**:
  - "build model"
  - "check and resume a model"
  - "half-scale" resizing used for multi-scale feature extraction
  - "transform to tensor" - converts PIL images to PyTorch tensors
  - "save features" - stores extracted features as numpy arrays
- **Note**: Missing normalization applied in content_aware version (bug)

#### `demo_content_aware_feats.py`
- **Purpose**: Extract content-aware features from images  
- **Key Comments**:
  - "build model"
  - "check and resume a model"
  - "half-scale" resizing for multi-scale extraction
  - "transform to tensor"
  - Includes proper normalize transforms: Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - "save features"
- **Difference from quality_aware**: Includes ImageNet normalization

---

### Dataset Module (`datasets/`)

#### `dataset.py` - IQAImageClass

**Class**: `IQAImageClass(data.Dataset)`

**Purpose**: Creates image dataset with distortion-based augmentations for quality-aware contrastive learning

**Constructor Parameters**:
- `csv_path`: Path to CSV with image paths
- `n_aug=7`: Number of distortion augmentations to apply
- `n_scale=1`: Number of scales (1 or 2)
- `n_distortions=1`: Number of cascading distortions (sequential application)
- `patch_size=224`: Crop size for patches
- `swap_crops=1`: Whether to swap crop pairs for contrastive learning

**Key Methods & Logic**:

1. **`iqa_transformations(choice, im, level)`**
   - Applies one of 26 different distortion types
   - `choice`: distortion type (1-26)
   - `level`: severity (0-4)
   - Maps to functions: imblurgauss, imblurlens, imcolordiffuse, etc.
   - Comment: "level = random.randint(0,4)"

2. **Distortion Types (1-26)**:
   ```
   1. Gaussian Blur
   2. Lens Blur
   3. Color Diffusion
   4. Color Shift
   5. Color Saturation
   6. Saturation
   7. JPEG Compression
   8. Gaussian Noise
   9. Color Map Noise
   10. Impulse Noise
   11. Multiplicative Noise
   12. Denoising
   13. Brighten
   14. Darken
   15. Mean Shift
   16. Resize Distortion
   17. Sharpen (High)
   18. Contrast
   19. Color Block
   20. Pixelate
   21. Non-Eccentricity
   22. Jitter
   23. Resize Distortion (Bilinear)
   24. Resize Distortion (Nearest)
   25. Resize Distortion (Lanczos)
   26. Motion Blur
   ```

3. **`crop_transform(image, crop_size=224)`**
   - Crops image to patch_size
   - Uses CenterCrop if image smaller than required
   - Uses RandomCrop otherwise
   - Comment: "if image.shape[2] < crop_size or image.shape[3] < crop_size"

4. **`__getitem__` Method - Data Generation Pipeline**:
   - Comment: "create positive pair"
   - Generates `n_aug+1` versions (1 original + n_aug augmented)
   - Comment: "generate self.aug distortion-augmentations"
   - Supports cascading distortions: `n_distortions > 1`
   - Comment: "generate two random crops"
   - **XOR Swap Logic** (BUGGY):
     ```
     chunk1_1[0:self.swap] = chunk1_1[0:self.swap] + chunk1_2[0:self.swap]    ## x = x + y
     chunk1_2[0:self.swap] = chunk1_1[0:self.swap] - chunk1_2[0:self.swap]    ## y = x - y
     chunk1_1[0:self.swap] = chunk1_1[0:self.swap] - chunk1_2[0:self.swap]    ## x = x - y
     ```
   - Double-scale support: `n_scale=2` creates additional half-resolution version

5. **Overlapping Area (OLA) Parameters**:
   - `min_OLA = 0.10` (10% minimum overlap)
   - `max_OLA = 0.30` (30% maximum overlap)
   - Used to generate random crop positions

#### `iqa_distortions.py` - Image Distortion Functions

**Purpose**: Implements 26 different distortion effects for image degradation

**Core Distortion Functions**:

1. **Blur Operations**:
   - `imblurgauss(im, level)`: Gaussian blur, levels [0.1, 0.5, 1, 2, 5]
   - `imblurlens(im, level)`: Lens blur (circular aperture), levels [1, 2, 4, 6, 8]
     - **Bug Location**: Line with `rad**2` should be `radius**2`
   - `imblurmotion(im, level)`: Motion blur, kernel_size [12, 16, 20, 24, 28]

2. **Color Operations**:
   - `imcolordiffuse(im, level)`: Diffuses color in LAB space, levels [1, 3, 6, 8, 12]
   - `imcolorshift(im, level)`: Shifts color channels, levels [1, 3, 6, 8, 12]
   - `imcolorsaturate(im, level)`: Adjusts saturation, levels [0.4, 0.2, 0.1, 0, -0.4]
   - `imsaturate(im, level)`: HSV saturation adjustment

3. **Noise Operations**:
   - `imnoisegauss(im, level)`: Gaussian noise
   - `imnoisecolormap(im, level)`: Color-mapped noise
   - `imnoiseimpulse(im, level)`: Salt-and-pepper noise
   - `imnoisemultiplicative(im, level)`: Multiplicative noise

4. **JPEG/Compression**:
   - `imcompressjpeg(im, level)`: JPEG compression with quality [25, 30, 35, 40, 45]

5. **Other Distortions**:
   - `imdenoise(im, level)`: Removes detail via denoising
   - `imbrighten(im, level)`: Increases brightness
   - `imdarken(im, level)`: Decreases brightness
   - `immeanshift(im, level)`: Edge-preserving smoothing
   - `imresizedist(im, level)`: Resize + downsample
   - `imsharpenHi(im, level)`: High-frequency sharpening
   - `imcontrastc(im, level)`: Contrast adjustment
   - `imcolorblock(im, level)`: Blocks color information
   - `impixelate(im, level)`: Pixelation effect
   - `imnoneccentricity(im, level)`: Spatial distortion
   - `imjitter(im, level)`: Random spatial jitter

**Utility Functions**:
- `curvefit(xx, coef)`: Spline-based curve fitting
- `mapmm(e)`: Min-max normalization
- **Comment**: `warnings.filterwarnings("ignore")` - Suppresses all warnings

---

### Learning Module (`learning/`)

#### `base_trainer.py`

**Class**: `BaseTrainer`

**Purpose**: Base training utilities for DDP setup and learning rate scheduling

**Methods**:

1. **`init_ddp_environment(gpu, ngpus_per_node)`**
   - Initializes DDP for distributed training
   - Sets GPU device
   - Creates local groups for ShuffleBN within nodes
   - Args:
     - `gpu`: current GPU ID
     - `ngpus_per_node`: num processes/GPUs per node

2. **`init_tensorboard_logger()`**
   - Initializes TensorBoard logging on rank 0 only
   - Creates logger with auto-flush every 2 seconds

3. **`adjust_learning_rate(optimizer, epoch)`**
   - Supports cosine annealing: uses `eta_min = lr * (lr_decay_rate ** 3)`
   - Supports step-based decay at epochs in `lr_decay_epochs`
   - Updates all param groups

4. **`warmup_learning_rate(epoch, batch_id, total_batches, optimizer)`**
   - Linear warmup over first `warm_epochs`
   - Linearly interpolates from `warmup_from` to `warmup_to`
   - Comment: "p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)"

---

### Memory & MoCo Module (`memory/`, `moco/`)

#### `distortion_augmentations.py` - Specialist Head Augmentations

**Purpose**: Quality-aware distortion-specific augmentations for MultiDistortionEncoder

**Key Concept**:
- Standard MoCo uses general random cropping
- Quality-aware MoCo uses TARGETED augmentations per distortion type
- Each specialist head sees only its own distortion type during training

**Contrastive Learning Strategy**:
- **Standard**: view1, view2 = two random crops → similar embeddings
- **Quality-aware**: 
  - anchor = clean/mild image
  - positive = similar distortion level
  - negative = much higher distortion
  - → position in embedding space = severity

**ARNIQA Manifold Extension**:
- Trained implicitly through severity-based contrastive signal
- After training, Gaussian head embedding space forms smooth severity curve

**Base Transforms** (applied AFTER distortion):
- `RandomResizedCrop(224, scale=(0.2, 1.0))`
- `RandomHorizontalFlip()`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

**Distortion Functions** (severity: 0.0 = mild, 1.0 = heavy):
- `apply_gaussian_noise`: sigma range [2, 60]
- `apply_blur`: radius range [0.3, 8.0]
- `apply_jpeg_compression`: quality range [5, 95]
- `apply_weather_haze`: alpha (blend factor) range [0.05, 0.70]

**DISTORTION_FN Registry**:
```python
{
    'gaussian': apply_gaussian_noise,
    'blur':     apply_blur,
    'jpeg':     apply_jpeg_compression,
    'weather':  apply_weather_haze,
}
```

---

### Distortion Augmentation Commented Code

**File**: `iqa_distortions.py` - Commented motion blur implementation:
- Original complex MATLAB-ported version calculating motion kernel
- Coefficients: `[1, 2, 4, 6, 8]` for radius
- Simplified version uses hardcoded horizontal/vertical kernels of size [12, 16, 20, 24, 28]

**File**: `dataset.py` - Commented multi-scale training code:
- Multi-scale version for double resolution features
- Downsampling via `F.interpolate(..., mode='bicubic')`
- Would add ~50-100 lines per epoch if enabled

---

## Data Pipeline & Distortions

### Image Degradation Chain
1. Load original image from CSV path
2. Apply 1 of 26 distortions at random level (0-4)
3. Optional: chain 2 distortions (first at level L1, second at level L2)
4. Crop to patches with controlled overlap (10-30%)
5. Optionally swap crop pairs between images
6. Optional: downsample to half-resolution for multi-scale

### OLA (Overlapping Area) Calculation
```
min_OLA = 0.10  (10% minimum)
max_OLA = 0.30  (30% maximum)

Random crop position:
  y ∈ [patch_size * (1-max_OLA), patch_size * (1-min_OLA)]
  x ∈ [(patch_size² * (1-max_OLA) - patch_size*y) / (patch_size-y), ...]
```

---

## Training Architecture

### DDP Training Setup (main_contrast.py)

**Multi-Process Spawning**:
- If multiprocessing_distributed: spawn ngpus_per_node processes
- Each process handles 1 GPU

**Core Components**:
1. Model + Model EMA (exponential moving average)
2. Training dataset + DataLoader
3. Contrast memory bank
4. Loss function: CrossEntropyLoss
5. Optimizer: SGD, AdamW, or LARS selectable

**Training Loop**:
```
for epoch in range(start_epoch, epochs+1):
    sampler.set_epoch(epoch)  # Ensures different order each epoch
    adjust_learning_rate(optimizer, epoch)
    train_loss = trainer.train(epoch, loader, model, model_ema, ...)
    log_to_tensorboard(epoch, loss, learning_rate)
    save_checkpoint(model, model_ema, epoch)
```

**Optimizer Options**:
- `"SGD"`: Standard SGD with momentum
- `"AdamW"`: Adam with weight decay
- `"LARS"`: Large-batch LARS optimizer for synchronous training

---

## Network Components

### Build Backbone
- Loads ResNet or ResNest architectures
- Optional: pre-training weights
- Creates feature encoder

### Build Linear
- Adds linear evaluation head
- Used for downstream task evaluation

### Multi-Distortion Encoder (MDE)
- Multiple specialist heads, one per distortion type
- Shared trunk encoder
- Each head trained on its own distortion

---

## Configuration & Options

### TrainOptions (base_options.py + train_options.py)
**Key Training Configuration**:
- `world_size`: Number of nodes × GPUs per node
- `rank`: Global process rank
- `multiprocessing_distributed`: DDP mode flag
- `dist_backend`: "nccl" or "gloo"
- `dist_url`: Master node address for DDP

**Learning Hyperparameters**:
- `learning_rate`: Base LR
- `momentum`: SGD momentum
- `weight_decay`: L2 regularization
- `optimizer`: Choice of SGD/AdamW/LARS
- `cosine`: Whether to use cosine annealing
- `warm_epochs`: Number of warmup epochs
- `warmup_from`, `warmup_to`: Warmup LR range
- `lr_decay_epochs`: Epochs to decay at
- `lr_decay_rate`: Multiplicative decay factor

**Model Architecture**:
- `arch`: Network architecture (resnet50, resnest50, etc.)
- `num_cluster`: Number of memory bank clusters
- `feat_dim`: Feature dimension

**Data Configuration**:
- `batch_size`: Samples per GPU
- `num_workers`: DataLoader workers
- `data_folder`: Root data directory
- `crop_low`: Low crop scale
- `crop_high`: High crop scale

---

## Constants & Magic Numbers

### Image Processing
| Value | Location | Purpose |
|-------|----------|---------|
| 0.10  | dataset.py | min_OLA (minimum overlapping area) |
| 0.30  | dataset.py | max_OLA (maximum overlapping area) |
| 224   | dataset.py | Default patch_size |
| 7     | dataset.py | Default n_aug (number of augmentations) |
| 26    | dataset.py | Number of distortion types |
| 4     | dataset.py | Severity levels (0-4) |

### Distortion Severity Levels
```python
# Gaussian Blur
levels = [0.1, 0.5, 1, 2, 5]

# Lens Blur
levels = [1, 2, 4, 6, 8]

# Motion Blur
levels = [12, 16, 20, 24, 28]

# Color operations
levels = [1, 3, 6, 8, 12] or [0.4, 0.2, 0.1, 0, -0.4]

# JPEG Compression
levels = [25, 30, 35, 40, 45]
```

### ImageNet Normalization
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## Checkpoint & Storage

### Standard Checkpoint Keys
```python
checkpoint = {
    'model': model.state_dict(),
    'model_ema': model_ema.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}
```

### Feature Extraction Output
- Numpy arrays (.npy format)
- Shape: (batch_size, feature_dim)
- Naming: `{image_name}_quality_aware_features.npy`

### TensorBoard Logging
- Logs: training loss, learning rate, gradients
- Output directory: `args.tb_folder`
- Auto-flush: every 2 seconds

---

## Important Implementation Notes

### Bug Alerts 🐛

1. **imblurlens.py Line ~59**: Uses undefined `rad` instead of `radius`
2. **dataset.py Lines 273-276**: XOR swap reuses modified values
3. **demo_quality_aware_feats.py**: Missing ImageNet normalization (inconsistent with content-aware)
4. **distortion_augmentations.py Line 135**: BytesIO buffer may be garbage collected

### Deprecated/Suppressed Warnings

- All warnings suppressed with `warnings.filterwarnings("ignore")`
- This masks potentially important DeprecationWarnings
- Recommendation: Only suppress specific warnings

### Missing Error Handling

- No file existence checks before torch.load()
- No try-except around Image.open()
- No validation of CSV file structure
- No permission checks for makedirs()

---

## References

- **Re-IQA Paper**: Original quality-aware contrastive learning
- **MoCo**: Momentum Contrast (He et al., CVPR 2020)
- **ARNIQA**: Manifold learning for quality estimation

---

**Document Generated**: During code cleanup  
**Last Updated**: Code review phase  
**Maintainers**: REIQA Team
