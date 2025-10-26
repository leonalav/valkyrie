#!/bin/bash
set -euo pipefail

# Production Validation Runner Script
# Executes comprehensive BigBird+S5+HRM pipeline validation

echo "=========================================="
echo "BigBird+S5+HRM Production Validation"
echo "=========================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_SCRIPT="${SCRIPT_DIR}/production_validation.py"
OUTPUT_DIR="${SCRIPT_DIR}/validation_artifacts_$(date +%Y%m%d_%H%M%S)"
SEED=42
MAX_STEPS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--output-dir DIR] [--seed SEED] [--max-steps STEPS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR    Output directory for validation artifacts"
            echo "  --seed SEED         Global seed for deterministic execution (default: 42)"
            echo "  --max-steps STEPS   Maximum training steps for validation (default: 100)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Environment setup
echo "Setting up environment..."
echo "Working directory: ${SCRIPT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Seed: ${SEED}"
echo "Max steps: ${MAX_STEPS}"

# Check dependencies
echo "Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# Check required files
REQUIRED_FILES=(
    "src/model/__init__.py"
    "src/train/__init__.py"
    "src/io/__init__.py"
    "src/evaluation/__init__.py"
    "src/sharding/__init__.py"
    "configs/valkyrie_base.yaml"
    "configs/bigbird_s5_hrm_1_2b.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${SCRIPT_DIR}/${file}" ]]; then
        echo "ERROR: Required file not found: ${file}"
        exit 1
    fi
done

echo "All dependencies found ✓"

# Set environment variables for deterministic execution
export JAX_ENABLE_X64=true
export JAX_DEBUG_NANS=true
export PYTHONHASHSEED=${SEED}
export JAX_PLATFORMS=cpu  # Force CPU for validation (can be overridden)

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run validation
echo ""
echo "Starting production validation..."
echo "This may take several minutes..."
echo ""

# Execute validation with proper error handling
if python3 "${VALIDATION_SCRIPT}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --max-steps "${MAX_STEPS}"; then
    
    echo ""
    echo "=========================================="
    echo "VALIDATION COMPLETED SUCCESSFULLY ✓"
    echo "=========================================="
    echo "Results available in: ${OUTPUT_DIR}"
    
    # Display summary if available
    if [[ -f "${OUTPUT_DIR}/validation_report.md" ]]; then
        echo ""
        echo "Validation Summary:"
        echo "------------------"
        head -20 "${OUTPUT_DIR}/validation_report.md"
        echo ""
        echo "Full report: ${OUTPUT_DIR}/validation_report.md"
    fi
    
    exit 0
else
    echo ""
    echo "=========================================="
    echo "VALIDATION FAILED ✗"
    echo "=========================================="
    echo "Check logs in: ${OUTPUT_DIR}"
    
    # Display error summary if available
    if [[ -f "${OUTPUT_DIR}/validation.log" ]]; then
        echo ""
        echo "Recent errors:"
        echo "-------------"
        tail -20 "${OUTPUT_DIR}/validation.log"
    fi
    
    exit 1
fi