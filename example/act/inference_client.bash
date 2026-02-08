#!/bin/bash
# Inference Client Script for ACT Policy
# Connects to a remote inference server for distributed inference
# Use with inference_act_server.bash on a GPU machine

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
CAMERA_INDEX="0"
CAMERA_NAME="front"
CAMERA_WIDTH="640"
CAMERA_HEIGHT="480"
CAMERA_FPS="30"
SERVER_HOST="192.168.100.146"
SERVER_PORT="8000"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║          ACT Policy - Inference Client Script             ║"
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
    echo "  --camera-index INDEX    Camera device index (default: 0)"
    echo "  --camera-name NAME      Camera name identifier (default: front)"
    echo "  --camera-width WIDTH    Camera width (default: 640)"
    echo "  --camera-height HEIGHT  Camera height (default: 480)"
    echo "  --camera-fps FPS        Camera FPS (default: 30)"
    echo "  --server-host HOST      Inference server hostname/IP (default: 192.168.100.146)"
    echo "  --server-port PORT      Inference server port (default: 8000)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --server-host 10.0.0.5 --server-port 8080"
    echo "  $0 --robot-port /dev/ttyUSB0 --camera-index 2"
    echo ""
    echo -e "${BLUE}Note:${NC}"
    echo "  Make sure the inference server is running on the remote machine."
    echo "  Start it with: ./inference_act_server.bash"
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
    echo -e "  ${YELLOW}Camera Index:${NC}    ${CAMERA_INDEX}"
    echo -e "  ${YELLOW}Camera Name:${NC}     ${CAMERA_NAME}"
    echo -e "  ${YELLOW}Resolution:${NC}      ${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
    echo -e "  ${YELLOW}Camera FPS:${NC}      ${CAMERA_FPS}"
    echo ""
    echo -e "${BLUE}Server Configuration:${NC}"
    echo -e "  ${YELLOW}Server Host:${NC}     ${SERVER_HOST}"
    echo -e "  ${YELLOW}Server Port:${NC}     ${SERVER_PORT}"
    echo -e "  ${YELLOW}Server URL:${NC}      http://${SERVER_HOST}:${SERVER_PORT}"
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
            echo -e "      1. Server is running (./inference_act_server.bash)"
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
            --camera-index)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --camera-index requires a value"
                    exit 1
                fi
                CAMERA_INDEX="$2"
                shift 2
                ;;
            --camera-name)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --camera-name requires a value"
                    exit 1
                fi
                CAMERA_NAME="$2"
                shift 2
                ;;
            --camera-width)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --camera-width requires a value"
                    exit 1
                fi
                CAMERA_WIDTH="$2"
                shift 2
                ;;
            --camera-height)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --camera-height requires a value"
                    exit 1
                fi
                CAMERA_HEIGHT="$2"
                shift 2
                ;;
            --camera-fps)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --camera-fps requires a value"
                    exit 1
                fi
                CAMERA_FPS="$2"
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
            *)
                echo -e "${RED}Error:${NC} Unknown option: $1"
                print_usage
                exit 1
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
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run policy/act/inference_client.py \
        --robot-port "${ROBOT_PORT}" \
        --robot-id "${ROBOT_ID}" \
        --camera-index "${CAMERA_INDEX}" \
        --camera-name "${CAMERA_NAME}" \
        --camera-width "${CAMERA_WIDTH}" \
        --camera-height "${CAMERA_HEIGHT}" \
        --camera-fps "${CAMERA_FPS}" \
        --server-host "${SERVER_HOST}" \
        --server-port "${SERVER_PORT}"
}

main "$@"
