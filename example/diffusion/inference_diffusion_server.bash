#!/bin/bash
# Inference Server Script for Diffusion Policy
# Runs the model on a GPU machine and serves predictions over HTTP
# Use with inference_diffusion_client.bash on the robot machine

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
CHECKPOINT="${CHECKPOINT:-jliu6718/diffusion-so101-place_brick}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
TASK="${TASK:-place_brick}"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║      Diffusion Policy - Inference Server Script           ║"
    echo "║          (Serves predictions over HTTP)                   ║"
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
    echo "  CHECKPOINT          Model checkpoint from HuggingFace Hub or local path"
    echo "                      Format: {username}/{policy}-{robot}-{task}"
    echo "                      (default: jliu6718/diffusion-so101-place_brick)"
    echo "  HOST                Server bind address (default: 0.0.0.0)"
    echo "  PORT                Server port (default: 8000)"
    echo "  DEVICE              Compute device: cuda, cpu, mps (default: cuda)"
    echo "  TASK                Task name for logging (default: place_brick)"
    echo ""
    echo -e "${BLUE}Example:${NC}"
    echo "  CHECKPOINT=myuser/diffusion-so101-pick_cube PORT=8080 $0"
    echo ""
    echo -e "${BLUE}Client Connection:${NC}"
    echo "  After starting this server, run the client on the robot machine:"
    echo "  SERVER_HOST=<this-machine-ip> ./inference_diffusion_client.bash"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Project Root:${NC}    ${PROJECT_ROOT}"
    echo ""
    echo -e "${BLUE}Model Configuration:${NC}"
    echo -e "  ${YELLOW}Checkpoint:${NC}      ${CHECKPOINT}"
    echo -e "  ${YELLOW}Device:${NC}          ${DEVICE}"
    echo -e "  ${YELLOW}Task:${NC}            ${TASK}"
    echo ""
    echo -e "${BLUE}Server Configuration:${NC}"
    echo -e "  ${YELLOW}Host:${NC}            ${HOST}"
    echo -e "  ${YELLOW}Port:${NC}            ${PORT}"
    echo -e "  ${YELLOW}Server URL:${NC}      http://${HOST}:${PORT}"
    echo ""

    # Show network interfaces for client connection
    echo -e "${BLUE}Network Interfaces (for client connection):${NC}"
    if command -v ip &> /dev/null; then
        ip -4 addr show | grep -E "inet [0-9]" | grep -v "127.0.0.1" | awk '{print "  " $2}' | cut -d'/' -f1 | sed 's/^/  /'
    elif command -v ifconfig &> /dev/null; then
        ifconfig | grep "inet " | grep -v "127.0.0.1" | awk '{print "  " $2}'
    else
        echo "  (Could not determine IP addresses)"
    fi
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

    # Check for GPU/device
    case "${DEVICE}" in
        cuda)
            if command -v nvidia-smi &> /dev/null; then
                echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected"
                nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | while read line; do
                    echo -e "      ${line}"
                done
                # Check CUDA availability
                if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
                    echo -e "  ${GREEN}✓${NC} PyTorch CUDA is available"
                else
                    echo -e "  ${YELLOW}⚠${NC} PyTorch CUDA may not be available"
                fi
            else
                echo -e "${RED}Error:${NC} DEVICE=cuda but no NVIDIA GPU detected"
                echo "  Either install NVIDIA drivers or set DEVICE=cpu"
                exit 1
            fi
            ;;
        mps)
            if [[ "$(uname)" == "Darwin" ]]; then
                echo -e "  ${GREEN}✓${NC} macOS detected, MPS should be available"
            else
                echo -e "${RED}Error:${NC} DEVICE=mps but not running on macOS"
                exit 1
            fi
            ;;
        cpu)
            echo -e "  ${YELLOW}⚠${NC} Running on CPU - inference will be slower"
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown device: ${DEVICE}"
            echo "  Supported devices: cuda, cpu, mps"
            exit 1
            ;;
    esac

    # Check if port is available
    if command -v ss &> /dev/null; then
        if ss -tuln | grep -q ":${PORT} "; then
            echo -e "  ${RED}✗${NC} Port ${PORT} is already in use"
            echo -e "    Processes using the port:"
            ss -tulnp | grep ":${PORT} " | sed 's/^/      /'
            echo -e "    Choose a different port with: PORT=<new-port> $0"
            exit 1
        else
            echo -e "  ${GREEN}✓${NC} Port ${PORT} is available"
        fi
    elif command -v netstat &> /dev/null; then
        if netstat -tuln | grep -q ":${PORT} "; then
            echo -e "  ${RED}✗${NC} Port ${PORT} is already in use"
            exit 1
        else
            echo -e "  ${GREEN}✓${NC} Port ${PORT} is available"
        fi
    fi

    echo ""
}

# Main execution
main() {
    print_banner

    # Parse arguments
    DRY_RUN=false
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
            --task)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --task requires a value"
                    exit 1
                fi
                TASK="$2"
                shift 2
                ;;
            *)
                # Collect extra arguments to pass through
                EXTRA_ARGS+=("$1")
                shift
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

    echo -e "${GREEN}Starting inference server...${NC}"
    echo -e "${CYAN}Server will be available at http://${HOST}:${PORT}${NC}"
    echo -e "${CYAN}Press Ctrl+C to stop${NC}"
    echo ""

    cd "${PROJECT_ROOT}"
    exec uv run policy/diffusion/inference_server.py \
        --checkpoint "${CHECKPOINT}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --device "${DEVICE}" \
        --task "${TASK}" \
        "${EXTRA_ARGS[@]}"
}

main "$@"
