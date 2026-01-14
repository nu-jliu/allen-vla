#!/bin/bash
# Calibration Script for Leader/Follower Arms
# Calibrates the specified arm and saves the calibration data

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

# Default configuration (can be overridden by environment variables)
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
LEADER_ID="${LEADER_ID:-my_leader}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-my_follower}"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║              Robot Arm Calibration Script                 ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 <arm> [OPTIONS]"
    echo ""
    echo -e "${BLUE}Arguments:${NC}"
    echo "  arm                 Arm type: so101_leader or so101_follower (required)"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  -h, --help          Show this help message"
    echo "  --dry-run           Show configuration without running"
    echo ""
    echo -e "${BLUE}Environment Variables:${NC}"
    echo "  LEADER_PORT         Leader arm serial port (default: /dev/ttyACM1)"
    echo "  LEADER_ID           Leader arm ID (default: my_leader)"
    echo "  FOLLOWER_PORT       Follower arm serial port (default: /dev/ttyACM0)"
    echo "  FOLLOWER_ID         Follower arm ID (default: my_follower)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 so101_leader     # Calibrate leader arm"
    echo "  $0 so101_follower   # Calibrate follower arm"
    echo "  LEADER_PORT=/dev/ttyUSB0 $0 so101_leader"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo ""
    echo -e "${BLUE}Selected Arm:${NC}"
    echo -e "  ${YELLOW}Arm Type:${NC}        ${ARM_TYPE}"
    echo -e "  ${YELLOW}Port:${NC}            ${ARM_PORT}"
    echo -e "  ${YELLOW}ID:${NC}              ${ARM_ID}"
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

    # Check port
    if [[ ! -e "${ARM_PORT}" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Port ${ARM_PORT} not found"
        echo -e "    Available serial ports:"
        ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | sed 's/^/      /' || echo "      No serial ports found"
    else
        echo -e "  ${GREEN}✓${NC} Port found: ${ARM_PORT}"
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

    # First argument must be arm type
    if [[ $# -lt 1 ]]; then
        echo -e "${RED}Error:${NC} Arm type is required"
        print_usage
        exit 1
    fi

    case $1 in
        so101_leader)
            ARM_TYPE="so101_leader"
            ARM_PORT="${LEADER_PORT}"
            ARM_ID="${LEADER_ID}"
            shift
            ;;
        so101_follower)
            ARM_TYPE="so101_follower"
            ARM_PORT="${FOLLOWER_PORT}"
            ARM_ID="${FOLLOWER_ID}"
            shift
            ;;
        *)
            echo -e "${RED}Error:${NC} Invalid arm type: $1"
            echo "  Valid options: so101_leader, so101_follower"
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

    echo -e "${GREEN}Starting calibration for ${ARM_TYPE} arm...${NC}"
    echo -e "${CYAN}Follow the on-screen instructions to calibrate the arm${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    if [[ "${ARM_TYPE}" == "so101_leader" ]]; then
        exec uv run lerobot-calibrate \
            --teleop.type "${ARM_TYPE}" \
            --teleop.port "${ARM_PORT}" \
            --teleop.id "${ARM_ID}"
    else
        exec uv run lerobot-calibrate \
            --robot.type "${ARM_TYPE}" \
            --robot.port "${ARM_PORT}" \
            --robot.id "${ARM_ID}"
    fi
}

main "$@"
