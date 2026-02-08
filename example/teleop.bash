#!/bin/bash
# Teleoperation Script for Leader/Follower Arms
# Controls the follower arm by mirroring the leader arm movements

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
FREQUENCY="30.0"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              Teleoperation Control Script                 ║"
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
    echo "  --leader-port PORT    Leader arm serial port (default: /dev/ttyACM1)"
    echo "  --leader-id ID        Leader arm ID (default: my_leader)"
    echo "  --follower-port PORT  Follower arm serial port (default: /dev/ttyACM0)"
    echo "  --follower-id ID      Follower arm ID (default: my_follower)"
    echo "  --frequency HZ        Control frequency in Hz (default: 30.0)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0                    # Run with defaults"
    echo "  $0 --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1"
    echo "  $0 --frequency 60.0"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo ""
    echo -e "${BLUE}Robot Configuration:${NC}"
    echo -e "  ${YELLOW}Leader Port:${NC}     ${LEADER_PORT}"
    echo -e "  ${YELLOW}Leader ID:${NC}       ${LEADER_ID}"
    echo -e "  ${YELLOW}Follower Port:${NC}   ${FOLLOWER_PORT}"
    echo -e "  ${YELLOW}Follower ID:${NC}     ${FOLLOWER_ID}"
    echo -e "  ${YELLOW}Frequency:${NC}       ${FREQUENCY} Hz"
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
            --frequency)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --frequency requires a value"
                    exit 1
                fi
                FREQUENCY="$2"
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

    echo -e "${GREEN}Starting teleoperation...${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run teleop/teleop.py \
        --leader-port "${LEADER_PORT}" \
        --leader-id "${LEADER_ID}" \
        --follower-port "${FOLLOWER_PORT}" \
        --follower-id "${FOLLOWER_ID}" \
        --frequency "${FREQUENCY}"
}

main "$@"
