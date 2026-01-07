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

# Default configuration (can be overridden by environment variables)
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-my_follower}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_NAME="${CAMERA_NAME:-front}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
SERVER_HOST="${SERVER_HOST:-192.168.100.146}"
SERVER_PORT="${SERVER_PORT:-8000}"

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
    echo "  -h, --help          Show this help message"
    echo "  --dry-run           Show configuration without running"
    echo "  --test-connection   Test server connectivity and exit"
    echo ""
    echo -e "${BLUE}Environment Variables:${NC}"
    echo "  ROBOT_PORT          Robot serial port (default: /dev/ttyACM0)"
    echo "  ROBOT_ID            Robot ID (default: my_follower)"
    echo "  CAMERA_INDEX        Camera device index (default: 0)"
    echo "  CAMERA_NAME         Camera name identifier (default: front)"
    echo "  CAMERA_WIDTH        Camera width (default: 640)"
    echo "  CAMERA_HEIGHT       Camera height (default: 480)"
    echo "  CAMERA_FPS          Camera FPS (default: 30)"
    echo "  SERVER_HOST         Inference server hostname/IP (default: 192.168.100.146)"
    echo "  SERVER_PORT         Inference server port (default: 8000)"
    echo ""
    echo -e "${BLUE}Example:${NC}"
    echo "  SERVER_HOST=10.0.0.5 SERVER_PORT=8080 $0"
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

    # Check if curl or wget is available
    if command -v curl &> /dev/null; then
        echo -e "  Testing connection to http://${SERVER_HOST}:${SERVER_PORT}..."
        if curl -s --connect-timeout 5 "http://${SERVER_HOST}:${SERVER_PORT}/health" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Server is reachable and responding"
            return 0
        elif curl -s --connect-timeout 5 "http://${SERVER_HOST}:${SERVER_PORT}/" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} Server is reachable (no health endpoint)"
            return 0
        else
            echo -e "  ${RED}✗${NC} Cannot connect to server at http://${SERVER_HOST}:${SERVER_PORT}"
            echo -e "    Please verify:"
            echo -e "      1. Server is running (./inference_act_server.bash)"
            echo -e "      2. Correct SERVER_HOST and SERVER_PORT"
            echo -e "      3. Network connectivity and firewall rules"
            return 1
        fi
    elif command -v nc &> /dev/null; then
        echo -e "  Testing TCP connection to ${SERVER_HOST}:${SERVER_PORT}..."
        if nc -z -w 5 "${SERVER_HOST}" "${SERVER_PORT}" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Server port is reachable"
            return 0
        else
            echo -e "  ${RED}✗${NC} Cannot connect to ${SERVER_HOST}:${SERVER_PORT}"
            return 1
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Neither curl nor nc available, skipping connection test"
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
