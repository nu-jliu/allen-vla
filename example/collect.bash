#!/bin/bash
# Data Collection Script for ACT or Diffusion Policy
# Reads camera streams from running MJPEG servers configured in camera.toml
# Dataset will be pushed to: {username}/{policy}-{robot}-{task}
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
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
LEADER_PORT="/dev/ttyACM1"
LEADER_ID="my_leader"
FOLLOWER_PORT="/dev/ttyACM0"
FOLLOWER_ID="my_follower"
USERNAME="jliu6718"
ROBOT_TYPE="so101"
TASK="place_brick"
HZ="30"
CAMERA_CONFIG="${PROJECT_ROOT}/config/camera.toml"
DATA_ROOT="${PROJECT_ROOT}/data"
SERVER_PORT="1234"
PUSH_TO_HUB=true

# Policy type (will be set from argument)
POLICY_TYPE=""

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║            Policy - Data Collection Script                ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 <policy> [OPTIONS]"
    echo ""
    echo -e "${BLUE}Arguments:${NC}"
    echo "  policy              Policy type: act or diffusion (required)"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help            Show this help message"
    echo "  --dry-run             Show configuration without running"
    echo "  --task TASK           Task name (default: place_brick)"
    echo "  --leader-port PORT    Leader arm serial port (default: /dev/ttyACM1)"
    echo "  --leader-id ID        Leader arm ID (default: my_leader)"
    echo "  --follower-port PORT  Follower arm serial port (default: /dev/ttyACM0)"
    echo "  --follower-id ID      Follower arm ID (default: my_follower)"
    echo "  --username USER       HuggingFace username (default: jliu6718)"
    echo "  --robot-type TYPE     Robot type (default: so101)"
    echo "  --hz HZ               Collection frequency in Hz (default: 30)"
    echo "  --camera-config PATH  Camera config TOML file (default: config/camera.toml)"
    echo "  --data-root DIR       Data storage root (default: \$PROJECT_ROOT/data)"
    echo "  --server-port PORT    Server port (default: 1234)"
    echo "  --push-to-hub         Push to HuggingFace Hub (default)"
    echo "  --no-push-to-hub      Don't push to HuggingFace Hub"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 act --task pick_cube"
    echo "  $0 diffusion --username myuser --task pick_cube"
    echo "  $0 act --camera-config config/multi_camera.toml"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo -e "  ${YELLOW}Data Root:${NC}       ${DATA_ROOT}"
    echo ""
    echo -e "${BLUE}Robot Configuration:${NC}"
    echo -e "  ${YELLOW}Leader Port:${NC}     ${LEADER_PORT}"
    echo -e "  ${YELLOW}Leader ID:${NC}       ${LEADER_ID}"
    echo -e "  ${YELLOW}Follower Port:${NC}   ${FOLLOWER_PORT}"
    echo -e "  ${YELLOW}Follower ID:${NC}     ${FOLLOWER_ID}"
    echo -e "  ${YELLOW}Robot Type:${NC}      ${ROBOT_TYPE}"
    echo ""
    echo -e "${BLUE}Camera Configuration:${NC}"
    echo -e "  ${YELLOW}Config File:${NC}     ${CAMERA_CONFIG}"
    echo ""
    echo -e "${BLUE}Collection Settings:${NC}"
    echo -e "  ${YELLOW}Frequency:${NC}       ${HZ} Hz"
    echo -e "  ${YELLOW}Policy Type:${NC}     ${POLICY_TYPE}"
    echo -e "  ${YELLOW}Task:${NC}            ${TASK}"
    echo -e "  ${YELLOW}Server Port:${NC}     ${SERVER_PORT}"
    echo ""
    echo -e "${BLUE}HuggingFace Hub:${NC}"
    echo -e "  ${YELLOW}Username:${NC}        ${USERNAME}"
    echo -e "  ${YELLOW}Dataset Repo:${NC}    ${USERNAME}/${POLICY_TYPE}-${ROBOT_TYPE}-${TASK}"
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

    # Check leader port
    if [[ ! -e "${LEADER_PORT}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Leader port ${LEADER_PORT} not found"
        echo -e "    Available serial ports:"
        ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sed 's/^/      /' || echo "      No serial ports found"
    else
        echo -e "  ${GREEN}✓${NC} Leader port found: ${LEADER_PORT}"
    fi

    # Check follower port
    if [[ ! -e "${FOLLOWER_PORT}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Follower port ${FOLLOWER_PORT} not found"
    else
        echo -e "  ${GREEN}✓${NC} Follower port found: ${FOLLOWER_PORT}"
    fi

    # Check camera config
    if [[ ! -f "${CAMERA_CONFIG}" ]]; then
        echo -e "  ${RED}✗${NC} Camera config not found: ${CAMERA_CONFIG}"
    else
        echo -e "  ${GREEN}✓${NC} Camera config found: ${CAMERA_CONFIG}"
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

    # Check for help flag first
    for arg in "$@"; do
        case $arg in
            -h|--help)
                print_usage
                exit 0
                ;;
        esac
    done

    # First argument must be policy type
    if [[ $# -lt 1 ]]; then
        echo -e "${RED}Error:${NC} Policy type is required"
        print_usage
        exit 1
    fi

    case $1 in
        act|diffusion|pi0)
            POLICY_TYPE="$1"
            shift
            ;;
        *)
            echo -e "${RED}Error:${NC} Invalid policy type: $1"
            echo "  Valid options: act, diffusion, pi0"
            exit 1
            ;;
    esac

    # Parse remaining arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
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
            --leader-port)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --leader-port requires a value"
                    exit 1
                fi
                LEADER_PORT="$2"
                shift 2
                ;;
            --leader-id)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --leader-id requires a value"
                    exit 1
                fi
                LEADER_ID="$2"
                shift 2
                ;;
            --follower-port)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --follower-port requires a value"
                    exit 1
                fi
                FOLLOWER_PORT="$2"
                shift 2
                ;;
            --follower-id)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --follower-id requires a value"
                    exit 1
                fi
                FOLLOWER_ID="$2"
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
            --hz)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --hz requires a value"
                    exit 1
                fi
                HZ="$2"
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
            --data-root)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --data-root requires a value"
                    exit 1
                fi
                DATA_ROOT="$2"
                shift 2
                ;;
            --server-port)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --server-port requires a value"
                    exit 1
                fi
                SERVER_PORT="$2"
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
            *)
                echo -e "${RED}Error:${NC} Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    print_config
    check_dependencies

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${YELLOW}Dry run mode - not executing${NC}"
        exit 0
    fi

    # Build push flag
    PUSH_FLAG=""
    if [[ "${PUSH_TO_HUB}" == "true" ]]; then
        PUSH_FLAG="--push"
    fi

    echo -e "${GREEN}Starting data collection...${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop collection${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run data_collection/collect.py \
        --leader-port "${LEADER_PORT}" \
        --leader-id "${LEADER_ID}" \
        --follower-port "${FOLLOWER_PORT}" \
        --follower-id "${FOLLOWER_ID}" \
        --username "${USERNAME}" \
        --policy-type "${POLICY_TYPE}" \
        --robot-type "${ROBOT_TYPE}" \
        --task "${TASK}" \
        --hz "${HZ}" \
        ${PUSH_FLAG} \
        --camera-config "${CAMERA_CONFIG}" \
        --root "${DATA_ROOT}" \
        --port "${SERVER_PORT}"
}

main "$@"
