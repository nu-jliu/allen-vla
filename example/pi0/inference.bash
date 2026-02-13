#!/bin/bash
# Inference Script for PI0 Policy (Standalone)
# Evaluation dataset will be pushed to: {username}/eval_{policy}-{robot}-{task}
# Example: jliu6718/eval_pi0-so101-place_brick

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
CHECKPOINT="jliu6718/pi0-so101-place_brick"
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="my_follower"
CAMERA_CONFIG="${PROJECT_ROOT}/config/camera.toml"
FPS="30"
USERNAME="jliu6718"
ROBOT_TYPE="so101"
TASK=""
DATA_ROOT="${PROJECT_ROOT}/data"
PUSH_TO_HUB=true
DISPLAY_VIDEO=false
NUM_EPISODES="1"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         PI0 Policy - Standalone Inference Script          ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help              Show this help message"
    echo "  --dry-run               Show configuration without running"
    echo "  --task TASK             Task name (required)"
    echo "  --checkpoint MODEL      Model checkpoint (default: jliu6718/pi0-so101-place_brick)"
    echo "  --robot-port PORT       Robot serial port (default: /dev/ttyACM0)"
    echo "  --robot-id ID           Robot ID (default: my_follower)"
    echo "  --camera-config PATH    Camera config TOML file (default: config/camera.toml)"
    echo "  --fps FPS               Inference FPS (default: 30)"
    echo "  --username USER         HuggingFace username (default: jliu6718)"
    echo "  --robot-type TYPE       Robot type (default: so101)"
    echo "  --data-root DIR         Data storage root (default: \$PROJECT_ROOT/data)"
    echo "  --push-to-hub           Push evaluation to HuggingFace Hub (default)"
    echo "  --no-push-to-hub        Don't push to HuggingFace Hub"
    echo "  --display-video         Display video feed"
    echo "  --episode N             Number of episodes to run (default: 1)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --checkpoint myuser/pi0-so101-pick_cube --task pick_cube"
    echo "  $0 --display-video --fps 15"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo -e "  ${YELLOW}Data Root:${NC}       ${DATA_ROOT}"
    echo ""
    echo -e "${BLUE}Model Configuration:${NC}"
    echo -e "  ${YELLOW}Checkpoint:${NC}      ${CHECKPOINT}"
    echo ""
    echo -e "${BLUE}Robot Configuration:${NC}"
    echo -e "  ${YELLOW}Robot Port:${NC}      ${ROBOT_PORT}"
    echo -e "  ${YELLOW}Robot ID:${NC}        ${ROBOT_ID}"
    echo -e "  ${YELLOW}Robot Type:${NC}      ${ROBOT_TYPE}"
    echo ""
    echo -e "${BLUE}Camera Configuration:${NC}"
    echo -e "  ${YELLOW}Config File:${NC}     ${CAMERA_CONFIG}"
    echo ""
    echo -e "${BLUE}Inference Settings:${NC}"
    echo -e "  ${YELLOW}Inference FPS:${NC}   ${FPS}"
    echo -e "  ${YELLOW}Num Episodes:${NC}    ${NUM_EPISODES}"
    echo -e "  ${YELLOW}Task:${NC}            ${TASK}"
    echo -e "  ${YELLOW}Display Video:${NC}   ${DISPLAY_VIDEO}"
    echo ""
    echo -e "${BLUE}HuggingFace Hub:${NC}"
    echo -e "  ${YELLOW}Username:${NC}        ${USERNAME}"
    echo -e "  ${YELLOW}Eval Repo:${NC}       ${USERNAME}/eval_pi0-${ROBOT_TYPE}-${TASK}"
    echo -e "  ${YELLOW}Push to Hub:${NC}     ${PUSH_TO_HUB}"
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

    # Check robot port
    if [[ ! -e "${ROBOT_PORT}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Robot port ${ROBOT_PORT} not found"
        echo -e "    Available serial ports:"
        ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sed 's/^/      /' || echo "      No serial ports found"
    else
        echo -e "  ${GREEN}✓${NC} Robot port found: ${ROBOT_PORT}"
    fi

    # Check camera config
    if [[ ! -f "${CAMERA_CONFIG}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Camera config not found: ${CAMERA_CONFIG}"
    else
        echo -e "  ${GREEN}✓${NC} Camera config found: ${CAMERA_CONFIG}"
    fi

    # Check for CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/      /'
    else
        echo -e "  ${YELLOW}⚠${NC} No NVIDIA GPU detected, inference may be slow"
    fi

    # Check data directory
    if [[ ! -d "${DATA_ROOT}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Data directory does not exist, will be created: ${DATA_ROOT}"
    else
        echo -e "  ${GREEN}✓${NC} Data directory exists: ${DATA_ROOT}"
    fi

    echo ""
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
            --checkpoint)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --checkpoint requires a value"
                    exit 1
                fi
                CHECKPOINT="$2"
                shift 2
                ;;
            --robot-port)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --robot-port requires a value"
                    exit 1
                fi
                ROBOT_PORT="$2"
                shift 2
                ;;
            --robot-id)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --robot-id requires a value"
                    exit 1
                fi
                ROBOT_ID="$2"
                shift 2
                ;;
            --camera-config)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --camera-config requires a value"
                    exit 1
                fi
                CAMERA_CONFIG="$2"
                shift 2
                ;;
            --fps)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --fps requires a value"
                    exit 1
                fi
                FPS="$2"
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
            --robot-type)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --robot-type requires a value"
                    exit 1
                fi
                ROBOT_TYPE="$2"
                shift 2
                ;;
            --data-root)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --data-root requires a value"
                    exit 1
                fi
                DATA_ROOT="$2"
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
            --display-video)
                DISPLAY_VIDEO=true
                shift
                ;;
            --episode)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --episode requires a value"
                    exit 1
                fi
                NUM_EPISODES="$2"
                shift 2
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

    # If task is specified, update CHECKPOINT to use it
    if [[ -n "${TASK}" && "${CHECKPOINT}" == *-* ]]; then
        # Extract username and policy-robot from CHECKPOINT, replace task
        CHECKPOINT=$(echo "${CHECKPOINT}" | sed "s/-[^-]*$/-${TASK}/")
    fi

    print_config
    check_dependencies

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${YELLOW}Dry run mode - not executing${NC}"
        exit 0
    fi

    # Build optional flags
    PUSH_FLAG=""
    if [[ "${PUSH_TO_HUB}" == "true" ]]; then
        PUSH_FLAG="--push-to-hub"
    fi

    DISPLAY_FLAG="--no-display"
    if [[ "${DISPLAY_VIDEO}" == "true" ]]; then
        DISPLAY_FLAG=""
    fi

    echo -e "${GREEN}Starting inference...${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop inference${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run policy/pi0/inference.py \
        --checkpoint "${CHECKPOINT}" \
        --robot-port "${ROBOT_PORT}" \
        --camera-config "${CAMERA_CONFIG}" \
        --username "${USERNAME}" \
        --robot-type "${ROBOT_TYPE}" \
        --task "${TASK}" \
        --robot-id "${ROBOT_ID}" \
        --fps "${FPS}" \
        --episode "${NUM_EPISODES}" \
        --root "${DATA_ROOT}" \
        ${PUSH_FLAG} \
        ${DISPLAY_FLAG}
}

main "$@"
