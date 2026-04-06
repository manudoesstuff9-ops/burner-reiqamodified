################################################################################
# RE-IQA COMPLETE PROJECT RUNNER - SINGLE SCRIPT
# 
# This script:
# 1. Activates virtual environment
# 2. Verifies Python/CUDA/dependencies
# 3. Creates required directories
# 4. Runs complete training pipeline
#
# Usage: ./RUN_PROJECT_COMPLETE.ps1
################################################################################

# Set error handling
$ErrorActionPreference = "Stop"

Write-Host "`n" + "="*80
Write-Host "RE-IQA COMPLETE PROJECT RUNNER"
Write-Host "="*80 + "`n"

# STEP 1: Activate Virtual Environment
Write-Host "[STEP 1/5] Activating virtual environment..." -ForegroundColor Cyan
try {
    & ".\.venv\Scripts\Activate.ps1"
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    & ".\.venv\Scripts\Activate.ps1"
    Write-Host "✓ New virtual environment created and activated" -ForegroundColor Green
}

# STEP 2: Verify Python and CUDA
Write-Host "`n[STEP 2/5] Verifying Python and CUDA..." -ForegroundColor Cyan
python -c "import torch; print(f'✓ PyTorch installed: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# STEP 3: Install/Update Requirements
Write-Host "`n[STEP 3/5] Installing/updating dependencies..." -ForegroundColor Cyan
python -m pip install -q -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# STEP 4: Create Required Directories
Write-Host "`n[STEP 4/5] Creating required directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "save_dir" | Out-Null
New-Item -ItemType Directory -Force -Path "tb_logger" | Out-Null
New-Item -ItemType Directory -Force -Path "csv_files" | Out-Null
New-Item -ItemType Directory -Force -Path "data/train" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# STEP 5: Run Training
Write-Host "`n[STEP 5/5] Starting training..." -ForegroundColor Cyan
Write-Host "`nRunning: python main_contrast.py" -ForegroundColor Yellow
Write-Host "Parameters:" -ForegroundColor Yellow
Write-Host "  - Batch size: 128" -ForegroundColor Yellow
Write-Host "  - Epochs: 200" -ForegroundColor Yellow
Write-Host "  - Learning rate: 0.05" -ForegroundColor Yellow
Write-Host "  - Optimizer: SGD" -ForegroundColor Yellow
Write-Host "  - Architecture: ResNet50" -ForegroundColor Yellow
Write-Host "  - Feature dimension: 256" -ForegroundColor Yellow
Write-Host "  - Cosine annealing: Enabled" -ForegroundColor Yellow
Write-Host "  - Multi-GPU: Auto-detect" -ForegroundColor Yellow
Write-Host "`nMake sure csv_files/moco_train.csv exists with image paths`n" -ForegroundColor Magenta

python main_contrast.py `
  --batch_size 128 `
  --epochs 200 `
  --learning_rate 0.05 `
  --optimizer SGD `
  --arch resnet50 `
  --feat_dim 256 `
  --cosine `
  --warm `
  --head linear `
  --num_workers 4 `
  --print_freq 10 `
  --save_freq 5 `
  --model_path ./save_dir `
  --tb_path ./tb_logger `
  --csv_path ./csv_files/moco_train.csv `
  --multiprocessing_distributed

Write-Host "`n" + "="*80
Write-Host "✓ TRAINING COMPLETE!"
Write-Host "="*80 + "`n"
Write-Host "Check results in:" -ForegroundColor Green
Write-Host "  - Checkpoints: ./save_dir/" -ForegroundColor Green
Write-Host "  - TensorBoard: ./tb_logger/" -ForegroundColor Green
Write-Host "`nTo view TensorBoard: tensorboard --logdir=./tb_logger/" -ForegroundColor Green
