#!/bin/bash

################################################################################
# RE-IQA COMPLETE SETUP & RUN - UNIVERSAL MASTER COMMAND (BASH VERSION)
# 
# Works on: Linux, macOS, Windows (WSL2)
#
# Usage: 
#   chmod +x MASTER_SETUP_AND_RUN.sh
#   ./MASTER_SETUP_AND_RUN.sh
#
# Or make it executable and run: bash MASTER_SETUP_AND_RUN.sh
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

PYTHON_VERSION="3.8"
BATCH_SIZE=128
EPOCHS=200
LEARNING_RATE=0.05
NUM_WORKERS=4
OPTIMIZER="SGD"
ARCH="resnet50"
FEAT_DIM=256
TRAINING_MODE="main"  # or "mde"

# ============================================================================
# COLORS FOR OUTPUT
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo -e "\n${CYAN}$(printf '=%.0s' {1..80})${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..80})${NC}\n"
}

print_step() {
    echo -e "${CYAN}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# MAIN SETUP FLOW
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_header "RE-IQA COMPLETE SETUP & RUN - MASTER COMMAND"

echo -e "Project Root: ${YELLOW}$PROJECT_ROOT${NC}"
echo -e "Training Mode: ${YELLOW}$TRAINING_MODE${NC}"
echo -e "Batch Size: ${YELLOW}$BATCH_SIZE${NC} | Epochs: ${YELLOW}$EPOCHS${NC}\n"

# ============================================================================
# STEP 1: CREATE/ACTIVATE VIRTUAL ENVIRONMENT
# ============================================================================

print_step "[STEP 1/7] Setting up Python virtual environment..."

VENV_PATH=".venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

echo "  Activating virtual environment..."
source "$VENV_PATH/bin/activate"

print_success "Virtual environment ready"

# ============================================================================
# STEP 2: UPGRADE PIP
# ============================================================================

print_step "[STEP 2/7] Upgrading pip..."
python -m pip install --upgrade pip -q
print_success "pip upgraded"

# ============================================================================
# STEP 3: INSTALL DEPENDENCIES
# ============================================================================

print_step "[STEP 3/7] Installing project dependencies..."

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

echo "  Installing packages from requirements.txt..."
python -m pip install -r requirements.txt -q

if [ $? -ne 0 ]; then
    print_warning "Installation had issues, retrying with cache disabled..."
    python -m pip install --no-cache-dir -r requirements.txt
fi

print_success "All dependencies installed"

# ============================================================================
# STEP 4: VERIFY INSTALLATION
# ============================================================================

print_step "[STEP 4/7] Verifying installation..."

python << 'EOF'
import torch
import sys

print('  Python version:', sys.version.split()[0])
print('  PyTorch version:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('  GPU device:', torch.cuda.get_device_name(0))
    print('  CUDA version:', torch.version.cuda)
else:
    print('  Using CPU mode (training will be slower)')

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

print('  \033[0;32m✓ All critical packages verified\033[0m')
EOF

# ============================================================================
# STEP 5: CREATE REQUIRED DIRECTORIES
# ============================================================================

print_step "[STEP 5/7] Creating directory structure..."

DIRECTORIES=(
    "save_dir"
    "save_dir/checkpoints"
    "tb_logger"
    "csv_files"
    "data"
    "data/train"
    "data/val"
    "features"
    "features/content"
    "features/quality"
    "logs"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  ✓ Created: $dir"
    else
        echo "  → Exists: $dir"
    fi
done

# ============================================================================
# STEP 6: PREPARE CSV IF NEEDED
# ============================================================================

print_step "[STEP 6/7] Checking CSV files..."

CSV_PATH="csv_files/moco_train.csv"

if [ ! -f "$CSV_PATH" ]; then
    print_warning "CSV file not found at: $CSV_PATH"
    echo "  Creating sample CSV template..."
    
    mkdir -p csv_files
    
    cat > "$CSV_PATH" << 'EOF'
# Sample CSV - Replace paths with your actual image paths
# Format: Image_path (one path per line)
sample_images/image1.jpg
sample_images/image2.jpg
sample_images/image3.jpg
EOF
    
    print_success "Sample CSV created at: $CSV_PATH"
    print_warning "IMPORTANT: Update CSV with actual image paths before training!"
else
    print_success "CSV file exists: $CSV_PATH"
fi

# ============================================================================
# STEP 7: RUN TRAINING
# ============================================================================

print_step "[STEP 7/7] Starting training pipeline..."

echo "────────────────────────────────────────────────────────────────────────────────"
echo -e "${YELLOW}TRAINING CONFIGURATION:${NC}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Optimizer: $OPTIMIZER"
echo "  Architecture: $ARCH"
echo "  Feature Dimension: $FEAT_DIM"
echo "  Workers: $NUM_WORKERS"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

if [ "$TRAINING_MODE" = "mde" ]; then
    echo -e "${CYAN}Running: main_contrast_mde.py (Multi-Distortion Encoder)${NC}"
    python main_contrast_mde.py \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --optimizer $OPTIMIZER \
        --arch $ARCH \
        --feat_dim $FEAT_DIM \
        --cosine \
        --warm \
        --head linear \
        --num_workers $NUM_WORKERS \
        --model_path ./save_dir \
        --tb_path ./tb_logger \
        --csv_path ./csv_files/moco_train.csv \
        --multiprocessing_distributed
else
    echo -e "${CYAN}Running: main_contrast.py (Original Re-IQA)${NC}"
    python main_contrast.py \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --optimizer $OPTIMIZER \
        --arch $ARCH \
        --feat_dim $FEAT_DIM \
        --cosine \
        --warm \
        --head linear \
        --num_workers $NUM_WORKERS \
        --model_path ./save_dir \
        --tb_path ./tb_logger \
        --csv_path ./csv_files/moco_train.csv \
        --multiprocessing_distributed
fi

EXIT_CODE=$?

# ============================================================================
# SUMMARY
# ============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    print_header "✓ TRAINING COMPLETED SUCCESSFULLY!"
    echo -e "Output locations:"
    echo -e "  Models: ${GREEN}./save_dir/${NC}"
    echo -e "  Logs: ${GREEN}./tb_logger/${NC}"
    echo -e "\nTo view TensorBoard:"
    echo -e "  ${YELLOW}tensorboard --logdir=./tb_logger${NC}"
    echo -e "  Then open: ${YELLOW}http://localhost:6006${NC}\n"
else
    print_header "✗ Training failed with exit code: $EXIT_CODE"
    exit 1
fi
