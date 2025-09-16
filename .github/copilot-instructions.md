# AI Agent Instructions for DDPM Implementation

## Project Overview
This repository implements a Denoising Diffusion Probabilistic Model (DDPM) with multi-platform GPU support, focusing on classroom/educational use. The implementation prioritizes clarity and minimal dependencies while maintaining performance.

## Key Architecture Components

### Core Model Structure
- **DDPM Class** (`DDPM` in `Classroom_DDPM_MPS_CUDA.ipynb`): Implements forward/reverse diffusion
- **TinyUNet** (`TinyUNet` class): Specialized U-Net architecture for 32x32 images
- **Diffusion Schedule** (`DiffusionSchedule` dataclass): Manages noise scheduling parameters

### Hardware Acceleration
```python
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA
    return torch.device("cpu")
```

## Development Workflows

### Environment Setup
1. Core dependencies: `torch`, `torchvision`, `matplotlib`
2. GPU support is automatically configured for:
   - Apple Silicon (MPS backend)
   - NVIDIA CUDA
   - CPU fallback

### Training Pipeline
1. Initialize with recommended parameters:
```python
train_ddpm(
    epochs=2,
    batch_size=128,
    lr=2e-4,
    timesteps=1000,
    base_channels=64,
    time_dim=256
)
```
2. Models save to `runs/ddpm.pt`
3. Sample images save to `runs/sample_step_*.png`

## Project Conventions

### Data Normalization
- CIFAR-10 images normalized to [-1, 1] range
- Always use provided `get_cifar10_loader()` for consistent preprocessing

### Model Architecture Patterns
- ResBlock + Up/Down modules for hierarchical feature extraction
- Time embeddings via sinusoidal positional encoding
- GroupNorm (8 groups) used throughout for stability

### Performance Considerations
- Batch size tuned for Apple Silicon memory constraints
- Uses `torch.set_float32_matmul_precision("high")` when available
- Gradient clipping norm = 1.0 for training stability

## Integration Points
- Data: CIFAR-10 via `torchvision.datasets`
- Visualization: `torchvision.utils` for image grid generation
- Checkpointing: Native PyTorch state dict saving/loading

## Common Tasks
- Train model: Run training cell with default parameters
- Generate samples: Use `sample_images()` after training
- Visualize noise schedule: Use `visualize_beta_schedule()`
- Monitor training: Check `runs/` directory for samples/checkpoints
