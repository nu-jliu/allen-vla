#!/bin/bash
# RealSense MJPEG Streaming Server
# Streams Intel RealSense camera video as MJPEG over HTTP for data collection

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
PORT="8080"
SERIAL=""
WIDTH="640"
HEIGHT="480"
FPS="30"
JPEG_QUALITY="80"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║        RealSense MJPEG Streaming Server                   ║"
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
    echo "  --port PORT             HTTP server port (default: 8080)"
    echo "  --serial SERIAL         RealSense serial number (default: first device)"
    echo "  --width WIDTH           Camera width (default: 640)"
    echo "  --height HEIGHT         Camera height (default: 480)"
    echo "  --fps FPS               Target FPS (default: 30)"
    echo "  --jpeg-quality QUALITY  JPEG quality 1-100 (default: 80)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0"
    echo "  $0 --port 8081 --serial 123456789"
    echo "  $0 --width 1280 --height 720 --fps 15"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo ""
    echo -e "${BLUE}Camera Settings:${NC}"
    echo -e "  ${YELLOW}Serial:${NC}          ${SERIAL:-"(first device)"}"
    echo -e "  ${YELLOW}Resolution:${NC}      ${WIDTH}x${HEIGHT}"
    echo -e "  ${YELLOW}FPS:${NC}             ${FPS}"
    echo -e "  ${YELLOW}JPEG Quality:${NC}    ${JPEG_QUALITY}"
    echo ""
    echo -e "${BLUE}Server Settings:${NC}"
    echo -e "  ${YELLOW}Port:${NC}            ${PORT}"
    echo -e "  ${YELLOW}Stream URL:${NC}      http://localhost:${PORT}/stream"
    echo -e "  ${YELLOW}Info URL:${NC}        http://localhost:${PORT}/info"
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
            --port)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --port requires a value"
                    exit 1
                fi
                PORT="$2"
                shift 2
                ;;
            --serial)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --serial requires a value"
                    exit 1
                fi
                SERIAL="$2"
                shift 2
                ;;
            --width)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --width requires a value"
                    exit 1
                fi
                WIDTH="$2"
                shift 2
                ;;
            --height)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --height requires a value"
                    exit 1
                fi
                HEIGHT="$2"
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
            --jpeg-quality)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --jpeg-quality requires a value"
                    exit 1
                fi
                JPEG_QUALITY="$2"
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

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${YELLOW}Dry run mode - not executing${NC}"
        exit 0
    fi

    # Build serial flag
    SERIAL_FLAG=""
    if [[ -n "${SERIAL}" ]]; then
        SERIAL_FLAG="--serial ${SERIAL}"
    fi

    echo -e "${GREEN}Starting RealSense MJPEG server...${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run data_collection/realsense.py \
        --port "${PORT}" \
        ${SERIAL_FLAG} \
        --width "${WIDTH}" \
        --height "${HEIGHT}" \
        --fps "${FPS}" \
        --jpeg-quality "${JPEG_QUALITY}"
}

main "$@"
