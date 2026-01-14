#!/bin/bash
# Training Script for ACT Policy
# Dataset/Model repo ID format: {username}/{policy}-{robot}-{task}
# Example: jliu6718/act-so101-place_brick

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
REPO_ID="${REPO_ID:-jliu6718/act-so101-place_brick}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/model}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10000}"
SEED="${SEED:-42}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
RESUME="${RESUME:-}"
DEVICE="${DEVICE:-cuda}"
TASK="${TASK:-}"
FORCE_REDOWNLOAD="${FORCE_REDOWNLOAD:-true}"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║             ACT Policy - Training Script                  ║"
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
    echo -e "${BLUE}Environment Variables:${NC}"
    echo "  REPO_ID             Dataset/Model repo ID on HuggingFace Hub"
    echo "                      Format: {username}/{policy}-{robot}-{task}"
    echo "                      (default: jliu6718/act-so101-place_brick)"
    echo "  OUTPUT_DIR          Local output directory for model checkpoints"
    echo "                      (default: \$PROJECT_ROOT/model)"
    echo "  BATCH_SIZE          Training batch size (default: 32)"
    echo "  STEPS               Number of training steps (default: 10000)"
    echo "  SEED                Random seed for reproducibility (default: 42)"
    echo "  PUSH_TO_HUB         Push trained model to HuggingFace Hub (default: true)"
    echo "  RESUME              Path to checkpoint to resume from (default: none)"
    echo "  DEVICE              Compute device: cuda, cpu, mps (default: cuda)"
    echo "  FORCE_REDOWNLOAD    Force re-download dataset from Hub (default: true)"
    echo ""
    echo -e "${BLUE}Example:${NC}"
    echo "  REPO_ID=myuser/act-so101-pick_cube STEPS=20000 $0"
    echo ""
    echo -e "${BLUE}Resume Training:${NC}"
    echo "  RESUME=/path/to/checkpoint $0"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo -e "  ${YELLOW}Output Dir:${NC}      ${OUTPUT_DIR}"
    echo ""
    echo -e "${BLUE}Dataset/Model:${NC}"
    echo -e "  ${YELLOW}Repo ID:${NC}         ${REPO_ID}"
    echo -e "  ${YELLOW}Push to Hub:${NC}     ${PUSH_TO_HUB}"
    echo -e "  ${YELLOW}Force Redownload:${NC} ${FORCE_REDOWNLOAD}"
    echo ""
    echo -e "${BLUE}Training Configuration:${NC}"
    echo -e "  ${YELLOW}Batch Size:${NC}      ${BATCH_SIZE}"
    echo -e "  ${YELLOW}Steps:${NC}           ${STEPS}"
    echo -e "  ${YELLOW}Seed:${NC}            ${SEED}"
    echo -e "  ${YELLOW}Device:${NC}          ${DEVICE}"
    if [[ -n "${RESUME}" ]]; then
        echo -e "  ${YELLOW}Resume From:${NC}     ${RESUME}"
    fi
    echo ""

    # Estimate training info
    echo -e "${BLUE}Training Estimate:${NC}"
    echo -e "  ${YELLOW}Total Steps:${NC}     ${STEPS}"
    echo -e "  ${YELLOW}Checkpoints:${NC}     Every 1000 steps (approx)"
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
                echo -e "${RED}Error:${NC} Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # If task is specified, update REPO_ID to use it
    if [[ -n "${TASK}" ]]; then
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

    # Build optional flags
    PUSH_FLAG=""
    if [[ "${PUSH_TO_HUB}" == "true" ]]; then
        PUSH_FLAG="--push"
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
        echo -e "${CYAN}Model will be pushed to: ${REPO_ID}${NC}"
    fi
    echo -e "${CYAN}Press Ctrl+C to interrupt (checkpoint will be saved)${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run policy/act/train.py \
        --repo-id "${REPO_ID}" \
        --output-dir "${OUTPUT_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        --steps "${STEPS}" \
        --seed "${SEED}" \
        ${PUSH_FLAG} \
        ${RESUME_FLAG} \
        ${FORCE_REDOWNLOAD_FLAG}
}

main "$@"
