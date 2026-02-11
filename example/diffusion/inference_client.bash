#!/bin/bash
# Inference Client Script for Diffusion Policy
# Connects to a remote inference server for distributed inference
# Use with inference_diffusion_server.bash on a GPU machine

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
ROBOT_PORT="/dev/ttyACM0"
ROBOT_ID="my_follower"
CAMERA_CONFIG="${PROJECT_ROOT}/config/camera.toml"
SERVER_HOST="192.168.100.146"
SERVER_PORT="8000"
FPS="30"
NUM_EPISODES="1"
EPISODE_TIME="60"
RESET_TIME="60"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║      Diffusion Policy - Inference Client Script           ║"
    echo "║        (Connects to remote inference server)              ║"
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
    echo "  --test-connection       Test server connectivity and exit"
    echo "  --robot-port PORT       Robot serial port (default: /dev/ttyACM0)"
    echo "  --robot-id ID           Robot ID (default: my_follower)"
    echo "  --camera-config PATH    Camera config TOML file (default: config/camera.toml)"
    echo "  --server-host HOST      Inference server hostname/IP (default: 192.168.100.146)"
    echo "  --server-port PORT      Inference server port (default: 8000)"
    echo "  --fps FPS               Control frequency in Hz (default: 30)"
    echo "  --episode N        Number of episodes (default: 10)"
    echo "  --episode-time SECS     Episode duration in seconds (default: 60)"
    echo "  --reset-time SECS       Reset time between episodes (default: 60)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --server-host 10.0.0.5 --server-port 8080"
    echo "  $0 --episode 5 --episode-time 30"
    echo ""
    echo -e "${BLUE}Note:${NC}"
    echo "  Make sure the inference server is running on the remote machine."
    echo "  Start it with: ./inference_diffusion_server.bash"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo ""
    echo -e "${BLUE}Robot Configuration:${NC}"
    echo -e "  ${YELLOW}Robot Port:${NC}      ${ROBOT_PORT}"
    echo -e "  ${YELLOW}Robot ID:${NC}        ${ROBOT_ID}"
    echo ""
    echo -e "${BLUE}Camera Configuration:${NC}"
    echo -e "  ${YELLOW}Config File:${NC}     ${CAMERA_CONFIG}"
    echo ""
    echo -e "${BLUE}Server Configuration:${NC}"
    echo -e "  ${YELLOW}Server Host:${NC}     ${SERVER_HOST}"
    echo -e "  ${YELLOW}Server Port:${NC}     ${SERVER_PORT}"
    echo -e "  ${YELLOW}Server URL:${NC}      http://${SERVER_HOST}:${SERVER_PORT}"
    echo ""
    echo -e "${BLUE}Episode Configuration:${NC}"
    echo -e "  ${YELLOW}FPS:${NC}             ${FPS}"
    echo -e "  ${YELLOW}Num Episodes:${NC}    ${NUM_EPISODES}"
    echo -e "  ${YELLOW}Episode Time:${NC}    ${EPISODE_TIME}s"
    echo -e "  ${YELLOW}Reset Time:${NC}      ${RESET_TIME}s"
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

    echo ""
}

# Test server connectivity
test_server_connection() {
    echo -e "${BLUE}Testing server connection...${NC}"

    # The inference server uses raw TCP sockets (not HTTP), so use nc for testing
    echo -e "  Testing TCP connection to ${SERVER_HOST}:${SERVER_PORT}..."
    if command -v nc &> /dev/null; then
        if nc -z -w 5 "${SERVER_HOST}" "${SERVER_PORT}" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Server port is reachable"
            return 0
        else
            echo -e "  ${RED}✗${NC} Cannot connect to ${SERVER_HOST}:${SERVER_PORT}"
            echo -e "    Please verify:"
            echo -e "      1. Server is running (./inference_diffusion_server.bash)"
            echo -e "      2. Correct --server-host and --server-port"
            echo -e "      3. Network connectivity and firewall rules"
            return 1
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} nc not available, skipping connection test"
        return 0
    fi
}

# Main execution
main() {
    print_banner

    # Parse arguments
    DRY_RUN=false
    TEST_CONNECTION=false
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
            --test-connection)
                TEST_CONNECTION=true
                shift
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
            --server-host)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --server-host requires a value"
                    exit 1
                fi
                SERVER_HOST="$2"
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
            --fps)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --fps requires a value"
                    exit 1
                fi
                FPS="$2"
                shift 2
                ;;
            --episode)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --episode requires a value"
                    exit 1
                fi
                NUM_EPISODES="$2"
                shift 2
                ;;
            --episode-time)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --episode-time requires a value"
                    exit 1
                fi
                EPISODE_TIME="$2"
                shift 2
                ;;
            --reset-time)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --reset-time requires a value"
                    exit 1
                fi
                RESET_TIME="$2"
                shift 2
                ;;
            *)
                # Collect extra arguments to pass through
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done

    print_config
    check_dependencies

    # Test connection
    if ! test_server_connection; then
        if [[ "${DRY_RUN}" != "true" ]]; then
            echo ""
            echo -e "${RED}Aborting due to connection failure.${NC}"
            echo -e "Use --dry-run to show configuration without connecting."
            exit 1
        fi
    fi

    if [[ "${TEST_CONNECTION}" == "true" ]]; then
        echo -e "${GREEN}Connection test completed.${NC}"
        exit 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${YELLOW}Dry run mode - not executing${NC}"
        exit 0
    fi

    echo -e "${GREEN}Starting inference client...${NC}"
    echo -e "${CYAN}Connecting to server at http://${SERVER_HOST}:${SERVER_PORT}${NC}"
    echo -e "${CYAN}Running ${NUM_EPISODES} episodes, ${EPISODE_TIME}s each${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run policy/diffusion/inference_client.py \
        --robot-port "${ROBOT_PORT}" \
        --robot-id "${ROBOT_ID}" \
        --camera-config "${CAMERA_CONFIG}" \
        --server-host "${SERVER_HOST}" \
        --server-port "${SERVER_PORT}" \
        --fps "${FPS}" \
        --episode "${NUM_EPISODES}" \
        --episode-time "${EPISODE_TIME}" \
        --reset-time "${RESET_TIME}" \
        "${EXTRA_ARGS[@]}"
}

main "$@"
