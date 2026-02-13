#!/bin/bash
# Rename HuggingFace Repo Script
# Downloads a HuggingFace repo to /tmp/ and pushes it to a new repo
# Example: ./rename-repo.bash --src user/old-repo --dst user/new-repo

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
SRC=""
DST=""
REPO_TYPE="dataset"
TMP_DIR="/tmp"

# Print banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║          HuggingFace Repo Rename Script                  ║"
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
    echo "  --src REPO            Source repo ID (e.g. user/old-repo) [required]"
    echo "  --dst REPO            Destination repo ID (e.g. user/new-repo) [required]"
    echo "  --type TYPE           Repo type: dataset or model (default: dataset)"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0 --src jliu6718/so101-place_brick --dst jliu6718/so101-pick_cube"
    echo "  $0 --src jliu6718/act-so101-place_brick --dst jliu6718/act-so101-pick_cube --type model"
}

# Print configuration
print_config() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  ${YELLOW}Source Repo:${NC}      ${SRC}"
    echo -e "  ${YELLOW}Destination Repo:${NC} ${DST}"
    echo -e "  ${YELLOW}Repo Type:${NC}        ${REPO_TYPE}"
    echo -e "  ${YELLOW}Download Dir:${NC}     ${TMP_DIR}"
    echo ""
}

# Check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    if ! command -v uv &> /dev/null; then
        echo -e "${RED}Error:${NC} 'uv' is not installed or not in PATH"
        echo "  Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} uv found"

    if ! uv run hf --help &> /dev/null; then
        echo -e "${RED}Error:${NC} 'hf' is not available via uv"
        echo "  Install with: uv add huggingface_hub[cli]"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} hf found (via uv)"

    if ! command -v git &> /dev/null; then
        echo -e "${RED}Error:${NC} 'git' is not installed or not in PATH"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} git found"

    if ! command -v git-lfs &> /dev/null; then
        echo -e "${RED}Error:${NC} 'git-lfs' is not installed or not in PATH"
        echo "  Install with: sudo apt install git-lfs"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} git-lfs found"

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

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --src)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --src requires a value"
                    exit 1
                fi
                SRC="$2"
                shift 2
                ;;
            --dst)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --dst requires a value"
                    exit 1
                fi
                DST="$2"
                shift 2
                ;;
            --type)
                if [[ -z "$2" || "$2" == --* ]]; then
                    echo -e "${RED}Error:${NC} --type requires a value"
                    exit 1
                fi
                REPO_TYPE="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}Error:${NC} Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "${SRC}" ]]; then
        echo -e "${RED}Error:${NC} --src is required"
        print_usage
        exit 1
    fi

    if [[ -z "${DST}" ]]; then
        echo -e "${RED}Error:${NC} --dst is required"
        print_usage
        exit 1
    fi

    if [[ "${REPO_TYPE}" != "dataset" && "${REPO_TYPE}" != "model" ]]; then
        echo -e "${RED}Error:${NC} --type must be 'dataset' or 'model'"
        exit 1
    fi

    print_config
    check_dependencies

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${YELLOW}Dry run mode - not executing${NC}"
        exit 0
    fi

    # Derive local path from source repo name
    LOCAL_DIR="${TMP_DIR}/${SRC##*/}"

    # Clean up any previous download
    if [[ -d "${LOCAL_DIR}" ]]; then
        echo -e "${YELLOW}Removing existing directory: ${LOCAL_DIR}${NC}"
        rm -rf "${LOCAL_DIR}"
    fi

    # Build type flag for hf cli
    TYPE_FLAG=""
    if [[ "${REPO_TYPE}" == "dataset" ]]; then
        TYPE_FLAG="--repo-type dataset"
    fi

    # Download the source repo
    echo -e "${GREEN}Downloading ${SRC} to ${LOCAL_DIR}...${NC}"
    uv run hf download "${SRC}" \
        ${TYPE_FLAG} \
        --local-dir "${LOCAL_DIR}"

    echo ""

    # Create the destination repo if it doesn't exist
    echo -e "${GREEN}Creating destination repo: ${DST}...${NC}"
    if uv run python -c "from huggingface_hub import repo_exists; exit(0 if repo_exists('${DST}', repo_type='${REPO_TYPE}') else 1)" 2>/dev/null; then
        echo -e "${YELLOW}Repo ${DST} already exists, skipping creation${NC}"
    else
        echo -e "${BLUE}Repo ${DST} not found, creating...${NC}"
        uv run hf repo create "${DST}" \
            --repo-type "${REPO_TYPE}" \
            || { echo -e "${RED}Failed to create repo ${DST}${NC}"; exit 1; }
        echo -e "${GREEN}Repo ${DST} created successfully${NC}"
    fi

    # Upload to the destination repo
    echo -e "${GREEN}Uploading to ${DST}...${NC}"
    uv run hf upload "${DST}" \
        "${LOCAL_DIR}" \
        ${TYPE_FLAG}

    echo ""

    # Clean up
    echo -e "${GREEN}Cleaning up ${LOCAL_DIR}...${NC}"
    rm -rf "${LOCAL_DIR}"

    echo ""
    echo -e "${GREEN}Done! Repo renamed from ${SRC} to ${DST}${NC}"
}

main "$@"
