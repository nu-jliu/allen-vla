#!/bin/bash
# Inference Script for ACT Policy (Standalone)
# Evaluation dataset will be pushed to: {username}/eval_{policy}-{robot}-{task}
# Example: jliu6718/eval_act-so101-place_brick

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
CHECKPOINT="${CHECKPOINT:-jliu6718/act-so101-place_brick}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_follower}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_NAME="${CAMERA_NAME:-front}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
FPS="${FPS:-30}"
USERNAME="${USERNAME:-jliu6718}"
POLICY_TYPE="${POLICY_TYPE:-act}"
ROBOT_TYPE="${ROBOT_TYPE:-so101}"
TASK="${TASK:-place_brick}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
DISPLAY_VIDEO="${DISPLAY_VIDEO:-false}"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         ACT Policy - Standalone Inference Script          ║"
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
    echo "  CHECKPOINT          Model checkpoint (default: jliu6718/act-so101-place_brick)"
    echo "  ROBOT_PORT          Robot serial port (default: /dev/ttyACM0)"
    echo "  ROBOT_ID            Robot ID (default: my_follower)"
    echo "  CAMERA_INDEX        Camera device index (default: 0)"
    echo "  CAMERA_NAME         Camera name identifier (default: front)"
    echo "  CAMERA_WIDTH        Camera width (default: 640)"
    echo "  CAMERA_HEIGHT       Camera height (default: 480)"
    echo "  CAMERA_FPS          Camera FPS (default: 30)"
    echo "  FPS                 Inference FPS (default: 30)"
    echo "  USERNAME            HuggingFace username (default: jliu6718)"
    echo "  POLICY_TYPE         Policy type (default: act)"
    echo "  ROBOT_TYPE          Robot type (default: so101)"
    echo "  TASK                Task name (default: place_brick)"
    echo "  DATA_ROOT           Data storage root (default: \$PROJECT_ROOT/data)"
    echo "  PUSH_TO_HUB         Push evaluation to HuggingFace Hub (default: true)"
    echo "  DISPLAY_VIDEO       Display video feed (default: false)"
    echo ""
    echo -e "${BLUE}Example:${NC}"
    echo "  CHECKPOINT=myuser/act-so101-pick_cube TASK=pick_cube $0"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo -e "  ${YELLOW}Data Root:${NC}       ${DATA_ROOT}"
    echo ""
    echo -e "${BLUE}Model Configuration:${NC}"
    echo -e "  ${YELLOW}Checkpoint:${NC}      ${CHECKPOINT}"
    echo -e "  ${YELLOW}Policy Type:${NC}     ${POLICY_TYPE}"
    echo ""
    echo -e "${BLUE}Robot Configuration:${NC}"
    echo -e "  ${YELLOW}Robot Port:${NC}      ${ROBOT_PORT}"
    echo -e "  ${YELLOW}Robot ID:${NC}        ${ROBOT_ID}"
    echo -e "  ${YELLOW}Robot Type:${NC}      ${ROBOT_TYPE}"
    echo ""
    echo -e "${BLUE}Camera Configuration:${NC}"
    echo -e "  ${YELLOW}Camera Index:${NC}    ${CAMERA_INDEX}"
    echo -e "  ${YELLOW}Camera Name:${NC}     ${CAMERA_NAME}"
    echo -e "  ${YELLOW}Resolution:${NC}      ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
    echo -e "  ${YELLOW}Camera FPS:${NC}      ${CAMERA_FPS}"
    echo ""
    echo -e "${BLUE}Inference Settings:${NC}"
    echo -e "  ${YELLOW}Inference FPS:${NC}   ${FPS}"
    echo -e "  ${YELLOW}Task:${NC}            ${TASK}"
    echo -e "  ${YELLOW}Display Video:${NC}   ${DISPLAY_VIDEO}"
    echo ""
    echo -e "${BLUE}HuggingFace Hub:${NC}"
    echo -e "  ${YELLOW}Username:${NC}        ${USERNAME}"
    echo -e "  ${YELLOW}Eval Repo:${NC}       ${USERNAME}/eval_${POLICY_TYPE}-${ROBOT_TYPE}-${TASK}"
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

    # Check camera
    if [[ ! -e "/dev/video${CAMERA_INDEX}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Camera /dev/video${CAMERA_INDEX} not found"
        echo -e "    Available cameras:"
        ls /dev/video* 2>/dev/null | sed 's/^/      /' || echo "      No cameras found"
    else
        echo -e "  ${GREEN}✓${NC} Camera found: /dev/video${CAMERA_INDEX}"
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
            *)
                echo -e "${RED}Error:${NC} Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

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
    exec uv run policy/act/inference.py \
        --checkpoint "${CHECKPOINT}" \
        --robot-port "${ROBOT_PORT}" \
        --camera-index "${CAMERA_INDEX}" \
        --username "${USERNAME}" \
        --policy-type "${POLICY_TYPE}" \
        --robot-type "${ROBOT_TYPE}" \
        --task "${TASK}" \
        --robot-id "${ROBOT_ID}" \
        --camera-name "${CAMERA_NAME}" \
        --camera-width "${CAMERA_WIDTH}" \
        --camera-height "${CAMERA_HEIGHT}" \
        --camera-fps "${CAMERA_FPS}" \
        --fps "${FPS}" \
        --root "${DATA_ROOT}" \
        ${PUSH_FLAG} \
        ${DISPLAY_FLAG}
}

main "$@"
