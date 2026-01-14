#!/bin/bash
# Training Script for Diffusion Policy
# Dataset/Model repo ID format: {username}/{policy}-{robot}-{task}
# Example: jliu6718/diffusion-so101-place_brick

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default configuration (can be overridden by environment variables)
REPO_ID="${REPO_ID:-jliu6718/diffusion-so101-place_brick}"
LOCAL_DIR="${LOCAL_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-100000}"
SEED="${SEED:-42}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
RESUME="${RESUME:-}"
DEVICE="${DEVICE:-cuda}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-true}"

# Diffusion-specific hyperparameters
HORIZON="${HORIZON:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
N_OBS_STEPS="${N_OBS_STEPS:-2}"
LR="${LR:-1e-4}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-250}"

# For push to hub
USERNAME="${USERNAME:-jliu6718}"
POLICY_TYPE="${POLICY_TYPE:-diffusion}"
ROBOT_TYPE="${ROBOT_TYPE:-so101}"
TASK="${TASK:-place_brick}"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║          Diffusion Policy - Training Script               ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help          Show this help message"
    echo "  --dry-run           Show configuration without running"
    echo "  --task TASK         Task name (overrides TASK env var)"
    echo ""
    echo -e "${BLUE}Environment Variables - Dataset:${NC}"
    echo "  REPO_ID             Dataset repo ID on HuggingFace Hub"
    echo "                      Format: {username}/{policy}-{robot}-{task}"
    echo "                      (default: jliu6718/diffusion-so101-place_brick)"
    echo "  LOCAL_DIR           Local dataset directory (overrides REPO_ID if set)"
    echo ""
    echo -e "${BLUE}Environment Variables - Training:${NC}"
    echo "  OUTPUT_DIR          Output directory for checkpoints (default: \$PROJECT_ROOT/model)"
    echo "  BATCH_SIZE          Training batch size (default: 8)"
    echo "  STEPS               Number of training steps (default: 100000)"
    echo "  SEED                Random seed for reproducibility (default: 42)"
    echo "  DEVICE              Compute device: cuda, cpu, mps (default: cuda)"
    echo "  RESUME              Path to checkpoint to resume from (default: none)"
    echo "  FORCE_REDOWNLOAD    Force re-download dataset from Hub (default: true)"
    echo ""
    echo -e "${BLUE}Environment Variables - Diffusion Hyperparameters:${NC}"
    echo "  HORIZON             Prediction horizon (default: 16)"
    echo "  N_ACTION_STEPS      Number of action steps (default: 8)"
    echo "  N_OBS_STEPS         Number of observation steps (default: 2)"
    echo "  LR                  Learning rate (default: 1e-4)"
    echo "  SAVE_FREQ           Checkpoint save frequency (default: 5000)"
    echo "  LOG_FREQ            Logging frequency (default: 250)"
    echo ""
    echo -e "${BLUE}Environment Variables - HuggingFace Hub:${NC}"
    echo "  PUSH_TO_HUB         Push trained model to HuggingFace Hub (default: true)"
    echo "  USERNAME            HuggingFace username (default: jliu6718)"
    echo "  POLICY_TYPE         Policy type (default: diffusion)"
    echo "  ROBOT_TYPE          Robot type (default: so101)"
    echo "  TASK                Task name (default: place_brick)"
    echo ""
    echo -e "${BLUE}Example:${NC}"
    echo "  REPO_ID=myuser/diffusion-so101-pick_cube STEPS=50000 $0"
    echo ""
    echo -e "${BLUE}Resume Training:${NC}"
    echo "  RESUME=/path/to/checkpoint $0"
    echo ""
    echo -e "${BLUE}Train from Local Dataset:${NC}"
    echo "  LOCAL_DIR=./data/my_dataset $0"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo -e "  ${YELLOW}Output Dir:${NC}      ${OUTPUT_DIR}"
    echo ""
    echo -e "${BLUE}Dataset:${NC}"
    if [[ -n "${LOCAL_DIR}" ]]; then
        echo -e "  ${YELLOW}Local Dir:${NC}       ${LOCAL_DIR}"
    else
        echo -e "  ${YELLOW}Repo ID:${NC}         ${REPO_ID}"
        echo -e "  ${YELLOW}Force Redownload:${NC} ${FORCE_REDOWNLOAD}"
    fi
    echo ""
    echo -e "${BLUE}Training Configuration:${NC}"
    echo -e "  ${YELLOW}Batch Size:${NC}      ${BATCH_SIZE}"
    echo -e "  ${YELLOW}Steps:${NC}           ${STEPS}"
    echo -e "  ${YELLOW}Seed:${NC}            ${SEED}"
    echo -e "  ${YELLOW}Device:${NC}          ${DEVICE}"
    echo -e "  ${YELLOW}Learning Rate:${NC}   ${LR}"
    if [[ -n "${RESUME}" ]]; then
        echo -e "  ${YELLOW}Resume From:${NC}     ${RESUME}"
    fi
    echo ""
    echo -e "${BLUE}Diffusion Hyperparameters:${NC}"
    echo -e "  ${YELLOW}Horizon:${NC}         ${HORIZON}"
    echo -e "  ${YELLOW}N Action Steps:${NC}  ${N_ACTION_STEPS}"
    echo -e "  ${YELLOW}N Obs Steps:${NC}     ${N_OBS_STEPS}"
    echo -e "  ${YELLOW}Save Freq:${NC}       ${SAVE_FREQ}"
    echo -e "  ${YELLOW}Log Freq:${NC}        ${LOG_FREQ}"
    echo ""
    echo -e "${BLUE}HuggingFace Hub:${NC}"
    echo -e "  ${YELLOW}Push to Hub:${NC}     ${PUSH_TO_HUB}"
    if [[ "${PUSH_TO_HUB}" == "true" ]]; then
        echo -e "  ${YELLOW}Username:${NC}        ${USERNAME}"
        echo -e "  ${YELLOW}Model Repo:${NC}      ${USERNAME}/${POLICY_TYPE}-${ROBOT_TYPE}-${TASK}"
    fi
    echo ""

    # Estimate training info
    echo -e "${BLUE}Training Estimate:${NC}"
    echo -e "  ${YELLOW}Total Steps:${NC}     ${STEPS}"
    echo -e "  ${YELLOW}Checkpoints:${NC}     Every ${SAVE_FREQ} steps"
    echo ""
}

# Check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    # Check for uv
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error:${NC} 'uv' is not installed or not in PATH"
        echo "  Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} uv found: $(uv --version)"

    # Check for GPU/device
    case "${DEVICE}" in
        cuda)
            if command -v nvidia-smi &> /dev/null; then
                echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected"
                nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | while read line; do
                    echo -e "      ${line}"
                done
                # Check CUDA availability
                if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                    echo -e "  ${GREEN}✓${NC} PyTorch CUDA is available"
                else
                    echo -e "  ${YELLOW}⚠${NC} PyTorch CUDA may not be available"
                fi
            else
                echo -e "${RED}Error:${NC} DEVICE=cuda but no NVIDIA GPU detected"
                echo "  Either install NVIDIA drivers or set DEVICE=cpu"
                exit 1
            fi
            ;;
        mps)
            if [[ "$(uname)" == "Darwin" ]]; then
                echo -e "  ${GREEN}✓${NC} macOS detected, MPS should be available"
            else
                echo -e "${RED}Error:${NC} DEVICE=mps but not running on macOS"
                exit 1
            fi
            ;;
        cpu)
            echo -e "  ${YELLOW}⚠${NC} Running on CPU - training will be very slow"
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown device: ${DEVICE}"
            echo "  Supported devices: cuda, cpu, mps"
            exit 1
            ;;
    esac

    # Check local directory if specified
    if [[ -n "${LOCAL_DIR}" ]]; then
        if [[ ! -d "${LOCAL_DIR}" ]]; then
            echo -e "  ${RED}✗${NC} Local directory not found: ${LOCAL_DIR}"
            exit 1
        else
            echo -e "  ${GREEN}✓${NC} Local directory found: ${LOCAL_DIR}"
        fi
    fi

    # Check output directory
    if [[ ! -d "${OUTPUT_DIR}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Output directory does not exist, will be created: ${OUTPUT_DIR}"
    else
        echo -e "  ${GREEN}✓${NC} Output directory exists: ${OUTPUT_DIR}"
        # Check for existing checkpoints
        if ls "${OUTPUT_DIR}"/*.pt 2>/dev/null | head -1 > /dev/null; then
            echo -e "      Existing checkpoints found:"
            ls -la "${OUTPUT_DIR}"/*.pt 2>/dev/null | tail -3 | sed 's/^/        /'
        fi
    fi

    # Check resume checkpoint if specified
    if [[ -n "${RESUME}" ]]; then
        if [[ ! -f "${RESUME}" ]]; then
            echo -e "  ${RED}✗${NC} Resume checkpoint not found: ${RESUME}"
            exit 1
        else
            echo -e "  ${GREEN}✓${NC} Resume checkpoint found: ${RESUME}"
        fi
    fi

    # Check disk space
    if command -v df &> /dev/null; then
        AVAILABLE_GB=$(df -BG "${OUTPUT_DIR}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
        if [[ -n "${AVAILABLE_GB}" ]] && [[ "${AVAILABLE_GB}" =~ ^[0-9]+$ ]]; then
            if [[ "${AVAILABLE_GB}" -lt 10 ]]; then
                echo -e "  ${YELLOW}⚠${NC} Low disk space: ${AVAILABLE_GB}GB available"
            else
                echo -e "  ${GREEN}✓${NC} Disk space: ${AVAILABLE_GB}GB available"
            fi
        fi
    fi

    echo ""
}

# Check HuggingFace authentication
check_hf_auth() {
    if [[ "${PUSH_TO_HUB}" == "true" ]]; then
        echo -e "${BLUE}Checking HuggingFace authentication...${NC}"
        if command -v huggingface-cli &> /dev/null; then
            if huggingface-cli whoami &> /dev/null; then
                HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
                echo -e "  ${GREEN}✓${NC} Logged in as: ${HF_USER}"
            else
                echo -e "  ${YELLOW}⚠${NC} Not logged in to HuggingFace Hub"
                echo "      Run: huggingface-cli login"
                echo "      Or set HF_TOKEN environment variable"
            fi
        elif [[ -n "${HF_TOKEN}" ]]; then
            echo -e "  ${GREEN}✓${NC} HF_TOKEN environment variable is set"
        else
            echo -e "  ${YELLOW}⚠${NC} Cannot verify HuggingFace authentication"
            echo "      Run: huggingface-cli login"
            echo "      Or set HF_TOKEN environment variable"
        fi
        echo ""
    fi
}

# Main execution
main() {
    print_banner

    # Parse arguments
    DRY_RUN=false
    EXTRA_ARGS=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --task)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --task requires a value"
                    exit 1
                fi
                TASK="$2"
                shift 2
                ;;
            *)
                # Collect extra arguments to pass through
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done

    # If task is specified, update REPO_ID to use it
    if [[ -n "${TASK}" && "${REPO_ID}" == *-* ]]; then
        # Extract username and policy-robot from REPO_ID, replace task
        REPO_ID=$(echo "${REPO_ID}" | sed "s/-[^-]*$/-${TASK}/")
    fi

    print_config
    check_dependencies
    check_hf_auth

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${YELLOW}Dry run mode - not executing${NC}"
        exit 0
    fi

    # Build dataset argument
    if [[ -n "${LOCAL_DIR}" ]]; then
        DATASET_ARG="--local-dir ${LOCAL_DIR}"
    else
        DATASET_ARG="--repo-id ${REPO_ID}"
    fi

    # Build optional flags
    PUSH_FLAG=""
    if [[ "${PUSH_TO_HUB}" == "true" ]]; then
        PUSH_FLAG="--push --username ${USERNAME} --policy-type ${POLICY_TYPE} --robot-type ${ROBOT_TYPE} --task ${TASK}"
    fi

    RESUME_FLAG=""
    if [[ -n "${RESUME}" ]]; then
        RESUME_FLAG="--resume ${RESUME}"
    fi

    FORCE_REDOWNLOAD_FLAG=""
    if [[ "${FORCE_REDOWNLOAD}" == "true" ]]; then
        FORCE_REDOWNLOAD_FLAG="--force-redownload"
    fi

    echo -e "${GREEN}Starting training...${NC}"
    echo -e "${CYAN}Training ${STEPS} steps with batch size ${BATCH_SIZE}${NC}"
    echo -e "${CYAN}Model will be saved to: ${OUTPUT_DIR}${NC}"
    if [[ "${PUSH_TO_HUB}" == "true" ]]; then
        echo -e "${CYAN}Model will be pushed to: ${USERNAME}/${POLICY_TYPE}-${ROBOT_TYPE}-${TASK}${NC}"
    fi
    echo -e "${CYAN}Press Ctrl+C to interrupt (checkpoint will be saved)${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run policy/diffusion/train.py \
        ${DATASET_ARG} \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        --steps "${STEPS}" \
        --horizon "${HORIZON}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --n-obs-steps "${N_OBS_STEPS}" \
        --lr "${LR}" \
        --save-freq "${SAVE_FREQ}" \
        --log-freq "${LOG_FREQ}" \
        --seed "${SEED}" \
        ${PUSH_FLAG} \
        ${RESUME_FLAG} \
        ${FORCE_REDOWNLOAD_FLAG} \
        "${EXTRA_ARGS[@]}"
}

main "$@"
