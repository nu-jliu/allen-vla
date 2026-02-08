#!/bin/bash
# Inference Client Script for Diffusion Policy (Quick Launch)
# Connects to a remote inference server for distributed inference

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
    echo "║   Diffusion Policy - Inference Client (Quick Launch)     ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help            Show this help message"
    echo "  --dry-run             Show configuration without running"
    echo "  --robot-port PORT     Robot serial port (default: /dev/ttyACM0)"
    echo "  --robot-id ID         Robot ID (default: my_follower)"
    echo "  --camera-index INDEX  Camera device index (default: 0)"
    echo "  --camera-name NAME    Camera name identifier (default: front)"
    echo "  --camera-width WIDTH  Camera width (default: 640)"
    echo "  --camera-height HEIGHT Camera height (default: 480)"
    echo "  --camera-fps FPS      Camera FPS (default: 30)"
    echo "  --server-host HOST    Inference server hostname/IP (default: 192.168.100.146)"
    echo "  --server-port PORT    Inference server port (default: 8000)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0                    # Run with defaults"
    echo "  $0 --server-host 10.0.0.5 --server-port 8080"
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

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${YELLOW}Dry run mode - not executing${NC}"
        exit 0
    fi

    echo -e "${GREEN}Starting inference client...${NC}"
    echo -e "${CYAN}Connecting to server at http://${SERVER_HOST}:${SERVER_PORT}${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run policy/diffusion/inference_client.py \
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
