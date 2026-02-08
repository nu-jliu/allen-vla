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

# Default configuration
REPO_ID="jliu6718/diffusion-so101-place_brick"
LOCAL_DIR=""
OUTPUT_DIR="${PROJECT_ROOT}/model"
BATCH_SIZE="32"
STEPS="100000"
SEED="42"
PUSH_TO_HUB=true
RESUME=""
DEVICE="cuda"
FORCE_REDOWNLOAD=true

# Diffusion-specific hyperparameters
HORIZON="16"
N_ACTION_STEPS="8"
N_OBS_STEPS="2"
LR="1e-4"
SAVE_FREQ="5000"
LOG_FREQ="250"

# For push to hub
USERNAME="jliu6718"
POLICY_TYPE="diffusion"
ROBOT_TYPE="so101"
TASK="place_brick"

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
    echo "  -h, --help                Show this help message"
    echo "  --dry-run                 Show configuration without running"
    echo "  --task TASK               Task name (updates repo ID suffix)"
    echo ""
    echo -e "${BLUE}Dataset Options:${NC}"
    echo "  --repo-id ID              Dataset repo ID (default: jliu6718/diffusion-so101-place_brick)"
    echo "  --local-dir DIR           Local dataset directory (overrides --repo-id)"
    echo "  --force-redownload        Force re-download dataset (default)"
    echo "  --no-force-redownload     Don't force re-download dataset"
    echo ""
    echo -e "${BLUE}Training Options:${NC}"
    echo "  --output-dir DIR          Output directory for checkpoints (default: \$PROJECT_ROOT/model)"
    echo "  --batch-size N            Training batch size (default: 32)"
    echo "  --steps N                 Number of training steps (default: 100000)"
    echo "  --seed N                  Random seed (default: 42)"
    echo "  --device DEVICE           Compute device: cuda, cpu, mps (default: cuda)"
    echo "  --resume PATH             Resume from checkpoint"
    echo ""
    echo -e "${BLUE}Diffusion Hyperparameters:${NC}"
    echo "  --horizon N               Prediction horizon (default: 16)"
    echo "  --n-action-steps N        Number of action steps (default: 8)"
    echo "  --n-obs-steps N           Number of observation steps (default: 2)"
    echo "  --lr RATE                 Learning rate (default: 1e-4)"
    echo "  --save-freq N             Checkpoint save frequency (default: 5000)"
    echo "  --log-freq N              Logging frequency (default: 250)"
    echo ""
    echo -e "${BLUE}HuggingFace Hub Options:${NC}"
    echo "  --push-to-hub             Push model to HuggingFace Hub (default)"
    echo "  --no-push-to-hub          Don't push to HuggingFace Hub"
    echo "  --username USER           HuggingFace username (default: jliu6718)"
    echo "  --policy-type TYPE        Policy type (default: diffusion)"
    echo "  --robot-type TYPE         Robot type (default: so101)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --repo-id myuser/diffusion-so101-pick_cube --steps 50000"
    echo "  $0 --task pick_cube"
    echo ""
    echo -e "${BLUE}Resume Training:${NC}"
    echo "  $0 --resume /path/to/checkpoint"
    echo ""
    echo -e "${BLUE}Train from Local Dataset:${NC}"
    echo "  $0 --local-dir ./data/my_dataset"
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
            --repo-id)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --repo-id requires a value"
                    exit 1
                fi
                REPO_ID="$2"
                shift 2
                ;;
            --local-dir)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --local-dir requires a value"
                    exit 1
                fi
                LOCAL_DIR="$2"
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
            --horizon)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --horizon requires a value"
                    exit 1
                fi
                HORIZON="$2"
                shift 2
                ;;
            --n-action-steps)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --n-action-steps requires a value"
                    exit 1
                fi
                N_ACTION_STEPS="$2"
                shift 2
                ;;
            --n-obs-steps)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --n-obs-steps requires a value"
                    exit 1
                fi
                N_OBS_STEPS="$2"
                shift 2
                ;;
            --lr)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --lr requires a value"
                    exit 1
                fi
                LR="$2"
                shift 2
                ;;
            --save-freq)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --save-freq requires a value"
                    exit 1
                fi
                SAVE_FREQ="$2"
                shift 2
                ;;
            --log-freq)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --log-freq requires a value"
                    exit 1
                fi
                LOG_FREQ="$2"
                shift 2
                ;;
            --username)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --username requires a value"
                    exit 1
                fi
                USERNAME="$2"
                shift 2
                ;;
            --policy-type)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --policy-type requires a value"
                    exit 1
                fi
                POLICY_TYPE="$2"
                shift 2
                ;;
            --robot-type)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --robot-type requires a value"
                    exit 1
                fi
                ROBOT_TYPE="$2"
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
