#!/bin/bash
# Training Script for PI0 Policy
# Dataset repo ID format: {username}/{robot}-{task}
# Example: jliu6718/so101-place_brick

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

# Default configuration
REPO_ID="jliu6718/so101-place_brick"
OUTPUT_DIR="${PROJECT_ROOT}/model"
BATCH_SIZE="8"
STEPS="10000"
SEED="42"
PUSH_TO_HUB=true
RESUME=""
DEVICE="cuda"
TASK=""
FORCE_REDOWNLOAD=true

# PI0-specific defaults
PALIGEMMA_VARIANT="gemma_2b"
ACTION_EXPERT_VARIANT="gemma_300m"
DTYPE="float32"
GRADIENT_CHECKPOINTING=false

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║             PI0 Policy - Training Script                  ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help                   Show this help message"
    echo "  --dry-run                    Show configuration without running"
    echo "  --task TASK                  Task name (required, updates repo ID suffix)"
    echo "  --repo-id ID                 Dataset repo ID (default: jliu6718/so101-place_brick)"
    echo "  --output-dir DIR             Output directory for checkpoints (default: \$PROJECT_ROOT/model)"
    echo "  --batch-size N               Training batch size (default: 8)"
    echo "  --steps N                    Number of training steps (default: 10000)"
    echo "  --seed N                     Random seed (default: 42)"
    echo "  --push-to-hub                Push model to HuggingFace Hub (default)"
    echo "  --no-push-to-hub             Don't push to HuggingFace Hub"
    echo "  --resume PATH                Resume from checkpoint"
    echo "  --device DEVICE              Compute device: cuda, cpu, mps (default: cuda)"
    echo "  --force-redownload           Force re-download dataset (default)"
    echo "  --no-force-redownload        Don't force re-download dataset"
    echo ""
    echo -e "${BLUE}PI0-specific Options:${NC}"
    echo "  --paligemma-variant VARIANT  PaliGemma variant: gemma_300m, gemma_2b (default: gemma_2b)"
    echo "  --action-expert-variant VAR  Action expert variant: gemma_300m, gemma_2b (default: gemma_300m)"
    echo "  --dtype TYPE                 Data type: bfloat16, float32 (default: float32)"
    echo "  --gradient-checkpointing     Enable gradient checkpointing"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --repo-id myuser/so101-pick_cube --steps 20000"
    echo "  $0 --task pick_cube"
    echo "  $0 --gradient-checkpointing --dtype bfloat16"
    echo ""
    echo -e "${BLUE}Resume Training:${NC}"
    echo "  $0 --resume /path/to/checkpoint"
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
    echo -e "${BLUE}PI0 Configuration:${NC}"
    echo -e "  ${YELLOW}PaliGemma Variant:${NC}       ${PALIGEMMA_VARIANT}"
    echo -e "  ${YELLOW}Action Expert Variant:${NC}   ${ACTION_EXPERT_VARIANT}"
    echo -e "  ${YELLOW}Dtype:${NC}                   ${DTYPE}"
    echo -e "  ${YELLOW}Gradient Checkpointing:${NC}  ${GRADIENT_CHECKPOINTING}"
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
        if uv run hf auth whoami &> /dev/null; then
            HF_USER=$(uv run hf auth whoami 2>/dev/null | head -1)
            echo -e "  ${GREEN}✓${NC} Logged in as: ${HF_USER}"
        elif [[ -n "${HF_TOKEN}" ]]; then
            echo -e "  ${GREEN}✓${NC} HF_TOKEN environment variable is set"
        else
            echo -e "  ${YELLOW}⚠${NC} Not logged in to HuggingFace Hub"
            echo "      Run: uv run hf auth login"
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
            --repo-id)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --repo-id requires a value"
                    exit 1
                fi
                REPO_ID="$2"
                shift 2
                ;;
            --output-dir)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --output-dir requires a value"
                    exit 1
                fi
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --batch-size)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --batch-size requires a value"
                    exit 1
                fi
                BATCH_SIZE="$2"
                shift 2
                ;;
            --steps)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --steps requires a value"
                    exit 1
                fi
                STEPS="$2"
                shift 2
                ;;
            --seed)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --seed requires a value"
                    exit 1
                fi
                SEED="$2"
                shift 2
                ;;
            --push-to-hub)
                PUSH_TO_HUB=true
                shift
                ;;
            --no-push-to-hub)
                PUSH_TO_HUB=false
                shift
                ;;
            --resume)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --resume requires a value"
                    exit 1
                fi
                RESUME="$2"
                shift 2
                ;;
            --device)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --device requires a value"
                    exit 1
                fi
                DEVICE="$2"
                shift 2
                ;;
            --force-redownload)
                FORCE_REDOWNLOAD=true
                shift
                ;;
            --no-force-redownload)
                FORCE_REDOWNLOAD=false
                shift
                ;;
            --paligemma-variant)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --paligemma-variant requires a value"
                    exit 1
                fi
                PALIGEMMA_VARIANT="$2"
                shift 2
                ;;
            --action-expert-variant)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --action-expert-variant requires a value"
                    exit 1
                fi
                ACTION_EXPERT_VARIANT="$2"
                shift 2
                ;;
            --dtype)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --dtype requires a value"
                    exit 1
                fi
                DTYPE="$2"
                shift 2
                ;;
            --gradient-checkpointing)
                GRADIENT_CHECKPOINTING=true
                shift
                ;;
            *)
                echo -e "${RED}Error:${NC} Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    if [[ -z "${TASK}" ]]; then
        echo -e "${RED}Error:${NC} --task is required"
        echo "  Example: $0 --task place_brick"
        exit 1
    fi

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

    GRADIENT_CHECKPOINTING_FLAG=""
    if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
        GRADIENT_CHECKPOINTING_FLAG="--gradient-checkpointing"
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
    exec uv run policy/pi0/train.py \
        --repo-id "${REPO_ID}" \
        --output-dir "${OUTPUT_DIR}" \
        --task "${TASK}" \
        --batch-size "${BATCH_SIZE}" \
        --steps "${STEPS}" \
        --seed "${SEED}" \
        --paligemma-variant "${PALIGEMMA_VARIANT}" \
        --action-expert-variant "${ACTION_EXPERT_VARIANT}" \
        --dtype "${DTYPE}" \
        ${PUSH_FLAG} \
        ${RESUME_FLAG} \
        ${FORCE_REDOWNLOAD_FLAG} \
        ${GRADIENT_CHECKPOINTING_FLAG}
}

main "$@"
