################################################################################
# RE-IQA COMPLETE SETUP & RUN - UNIVERSAL MASTER COMMAND
# 
# This script handles EVERYTHING:
# 1. Creates/activates virtual environment
# 2. Upgrades pip
# 3. Installs all dependencies
# 4. Verifies Python/CUDA/dependencies
# 5. Creates all required directories
# 6. Prepares data structure
# 7. Runs the training pipeline
#
# COPY & PASTE THIS ENTIRE SCRIPT INTO PowerShell
# Works on: Windows, Linux (WSL), macOS
#
# Usage: 
#   - Copy entire script to PowerShell
#   - Or save as .ps1 and run: .\MASTER_SETUP_AND_RUN.ps1
################################################################################

# ============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS IF NEEDED
# ============================================================================

$PYTHON_VERSION = "3.8"
$BATCH_SIZE = 128
$EPOCHS = 200
$LEARNING_RATE = 0.05
$NUM_WORKERS = 4
$OPTIMIZER = "SGD"
$ARCH = "resnet50"
$FEAT_DIM = 256

# Choose training mode:
# "main" for original Re-IQA
# "mde" for Multi-Distortion Encoder variant
$TRAINING_MODE = "main"

# ============================================================================
# STEP 1: BASIC SETUP & ERROR HANDLING
# ============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Get-Location

Write-Host "`n" + "="*80 -ForegroundColor Cyan
Write-Host "RE-IQA COMPLETE SETUP & RUN - MASTER COMMAND" -ForegroundColor Cyan
Write-Host "="*80 + "`n"
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Yellow
Write-Host "Training Mode: $TRAINING_MODE" -ForegroundColor Yellow
Write-Host "Batch Size: $BATCH_SIZE | Epochs: $EPOCHS | Learning Rate: $LEARNING_RATE" -ForegroundColor Yellow

# ============================================================================
# STEP 2: CREATE/ACTIVATE VIRTUAL ENVIRONMENT
# ============================================================================

Write-Host "`n[STEP 1/7] Setting up Python virtual environment..." -ForegroundColor Cyan

$VenvPath = ".\.venv"
$venvOrExists = Test-Path $VenvPath

if (-not $venvOrExists) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

Write-Host "  Activating virtual environment..." -ForegroundColor Yellow
& "$VenvPath\Scripts\Activate.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "  ✓ Virtual environment ready" -ForegroundColor Green

# ============================================================================
# STEP 3: UPGRADE PIP
# ============================================================================

Write-Host "`n[STEP 2/7] Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip -q
Write-Host "  ✓ pip upgraded" -ForegroundColor Green

# ============================================================================
# STEP 4: INSTALL DEPENDENCIES
# ============================================================================

Write-Host "`n[STEP 3/7] Installing project dependencies..." -ForegroundColor Cyan

if (-not (Test-Path "requirements.txt")) {
    Write-Host "  ✗ requirements.txt not found!" -ForegroundColor Red
    exit 1
}

Write-Host "  Installing packages from requirements.txt..." -ForegroundColor Yellow
python -m pip install -r requirements.txt -q

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
    Write-Host "  Trying with --no-cache-dir option..." -ForegroundColor Yellow
    python -m pip install --no-cache-dir -r requirements.txt
}

Write-Host "  ✓ All dependencies installed" -ForegroundColor Green

# ============================================================================
# STEP 5: VERIFY INSTALLATION
# ============================================================================

Write-Host "`n[STEP 4/7] Verifying installation..." -ForegroundColor Cyan

$VerifyScript = @"
import torch
import sys

print('  Python version:', sys.version.split()[0])
print('  PyTorch version:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('  GPU device:', torch.cuda.get_device_name(0))
    print('  CUDA version:', torch.version.cuda)
else:
    print('  Using CPU mode')

# Check other critical imports
try:
    import torchvision
    print('  TorchVision:', torchvision.__version__)
except:
    pass

try:
    import numpy
    print('  NumPy:', numpy.__version__)
except:
    pass

print('  ✓ All critical packages verified')
"@

python -c $VerifyScript

# ============================================================================
# STEP 6: CREATE REQUIRED DIRECTORIES
# ============================================================================

Write-Host "`n[STEP 5/7] Creating directory structure..." -ForegroundColor Cyan

$DirectoriesNeeded = @(
    "save_dir",
    "save_dir\checkpoints",
    "tb_logger",
    "csv_files",
    "data",
    "data\train",
    "data\val",
    "features",
    "features\content",
    "features\quality",
    "logs"
)

foreach ($dir in $DirectoriesNeeded) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "  ✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  → Exists: $dir" -ForegroundColor Gray
    }
}

# ============================================================================
# STEP 7: PREPARE CSV IF NEEDED
# ============================================================================

Write-Host "`n[STEP 6/7] Checking CSV files..." -ForegroundColor Cyan

$CsvPath = "csv_files\moco_train.csv"

if (-not (Test-Path $CsvPath)) {
    Write-Host "  ⚠ CSV file not found at: $CsvPath" -ForegroundColor Yellow
    Write-Host "  Creating sample CSV template..." -ForegroundColor Yellow
    
    New-Item -ItemType Directory -Force -Path "csv_files" | Out-Null
    
    $SampleCsv = @"
# Sample CSV - Replace paths with your actual image paths
# Format: Image_path (one path per line)
sample_images/image1.jpg
sample_images/image2.jpg
sample_images/image3.jpg
"@
    
    Set-Content -Path $CsvPath -Value $SampleCsv
    Write-Host "  ✓ Sample CSV created at: $CsvPath" -ForegroundColor Green
    Write-Host "  ⚠ IMPORTANT: Update CSV with actual image paths before training!" -ForegroundColor Yellow
} else {
    Write-Host "  ✓ CSV file exists: $CsvPath" -ForegroundColor Green
}

# ============================================================================
# STEP 8: RUN TRAINING
# ============================================================================

Write-Host "`n[STEP 7/7] Starting training pipeline..." -ForegroundColor Cyan
Write-Host "`n" + "-"*80
Write-Host "TRAINING CONFIGURATION:" -ForegroundColor Yellow
Write-Host "  Batch Size: $BATCH_SIZE" -ForegroundColor Gray
Write-Host "  Epochs: $EPOCHS" -ForegroundColor Gray
Write-Host "  Learning Rate: $LEARNING_RATE" -ForegroundColor Gray
Write-Host "  Optimizer: $OPTIMIZER" -ForegroundColor Gray
Write-Host "  Architecture: $ARCH" -ForegroundColor Gray
Write-Host "  Feature Dimension: $FEAT_DIM" -ForegroundColor Gray
Write-Host "  Workers: $NUM_WORKERS" -ForegroundColor Gray
Write-Host "-"*80 + "`n"

if ($TRAINING_MODE -eq "mde") {
    Write-Host "Running: main_contrast_mde.py (Multi-Distortion Encoder)" -ForegroundColor Cyan
    python main_contrast_mde.py `
        --batch_size $BATCH_SIZE `
        --epochs $EPOCHS `
        --learning_rate $LEARNING_RATE `
        --optimizer $OPTIMIZER `
        --arch $ARCH `
        --feat_dim $FEAT_DIM `
        --cosine `
        --warm `
        --head linear `
        --num_workers $NUM_WORKERS `
        --model_path ./save_dir `
        --tb_path ./tb_logger `
        --csv_path ./csv_files/moco_train.csv `
        --multiprocessing_distributed
} else {
    Write-Host "Running: main_contrast.py (Original Re-IQA)" -ForegroundColor Cyan
    python main_contrast.py `
        --batch_size $BATCH_SIZE `
        --epochs $EPOCHS `
        --learning_rate $LEARNING_RATE `
        --optimizer $OPTIMIZER `
        --arch $ARCH `
        --feat_dim $FEAT_DIM `
        --cosine `
        --warm `
        --head linear `
        --num_workers $NUM_WORKERS `
        --model_path ./save_dir `
        --tb_path ./tb_logger `
        --csv_path ./csv_files/moco_train.csv `
        --multiprocessing_distributed
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" + "="*80 -ForegroundColor Green
    Write-Host "✓ TRAINING COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "="*80
    Write-Host "`nOutput locations:" -ForegroundColor Green
    Write-Host "  Models: ./save_dir/" -ForegroundColor Green
    Write-Host "  Logs: ./tb_logger/" -ForegroundColor Green
    Write-Host "`nTo view TensorBoard:" -ForegroundColor Green
    Write-Host "  tensorboard --logdir=./tb_logger" -ForegroundColor Yellow
    Write-Host "  Then open: http://localhost:6006`n" -ForegroundColor Yellow
} else {
    Write-Host "`n" + "="*80 -ForegroundColor Red
    Write-Host "✗ Training failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "="*80 -ForegroundColor Red
    exit 1
}
