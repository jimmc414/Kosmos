#!/bin/bash
# Kosmos Production Verification Script
# Verifies all production requirements are met

set -e

echo "=== Kosmos Production Verification ==="
echo "Started at: $(date)"
echo ""

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
WARNINGS=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

# 1. Check Python version
echo "=== Step 1: Environment Check ==="
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
    pass "Python version: $PYTHON_VERSION (>= 3.11)"
else
    fail "Python version: $PYTHON_VERSION (requires >= 3.11)"
fi

# 2. Check package installation
echo ""
echo "=== Step 2: Package Import Check ==="
python3 -c "from kosmos.compression import ContextCompressor" 2>/dev/null && pass "Context compression module" || fail "Context compression module"
python3 -c "from kosmos.orchestration import ResearchOrchestrator" 2>/dev/null && pass "Orchestration module" || fail "Orchestration module"
python3 -c "from kosmos.validation import ScholarEvalValidator" 2>/dev/null && pass "Validation module" || fail "Validation module"
python3 -c "from kosmos.workflow import ResearchWorkflow" 2>/dev/null && pass "Workflow module" || fail "Workflow module"
python3 -c "from kosmos.execution import ProductionExecutor, PackageResolver" 2>/dev/null && pass "Execution module (new)" || fail "Execution module (new)"
python3 -c "from kosmos.monitoring import MetricsCollector" 2>/dev/null && pass "Monitoring module" || warn "Monitoring module (optional)"

# 3. Run smoke tests
echo ""
echo "=== Step 3: Smoke Tests ==="
if [ -f scripts/smoke_test.py ]; then
    python3 scripts/smoke_test.py && pass "Smoke tests" || fail "Smoke tests"
else
    warn "Smoke test script not found"
fi

# 4. Run core tests
echo ""
echo "=== Step 4: Core Tests ==="
pytest tests/unit/compression/ \
       tests/unit/orchestration/ \
       tests/unit/validation/ \
       tests/unit/workflow/ \
       tests/unit/agents/test_skill_loader.py \
       tests/unit/world_model/test_artifacts.py \
       -v --timeout=120 2>/dev/null && pass "Core tests" || fail "Core tests"

# 5. Run execution module tests
echo ""
echo "=== Step 5: Execution Module Tests ==="
pytest tests/unit/execution/test_package_resolver.py \
       tests/unit/execution/test_docker_manager.py \
       tests/unit/execution/test_production_executor.py \
       -v --timeout=60 2>/dev/null && pass "Execution module tests" || fail "Execution module tests"

# 6. Run integration tests
echo ""
echo "=== Step 6: Integration Tests ==="
pytest tests/integration/ tests/e2e/ -v --timeout=300 2>/dev/null && pass "Integration tests" || warn "Integration tests (some may need configuration)"

# 7. E2E verification
echo ""
echo "=== Step 7: E2E Verification ==="
if [ -f scripts/verify_e2e.py ]; then
    python3 scripts/verify_e2e.py --cycles 3 --tasks 5 && pass "E2E workflow verification" || warn "E2E verification (may need API keys)"
else
    warn "E2E verification script not found"
fi

# 8. Check Docker (optional)
echo ""
echo "=== Step 8: Docker Check (Optional) ==="
if command -v docker &> /dev/null; then
    docker --version && pass "Docker available"

    # Check if image exists or can be built
    if docker images | grep -q "kosmos-sandbox"; then
        pass "Sandbox image exists"
    else
        warn "Sandbox image not built (run: cd docker/sandbox && docker build -t kosmos-sandbox:latest .)"
    fi
else
    warn "Docker not installed (needed for sandboxed execution)"
fi

# 9. Check dependencies
echo ""
echo "=== Step 9: Dependency Check ==="
pip check 2>/dev/null && pass "No dependency conflicts" || warn "Some dependency conflicts found"

# 10. Summary
echo ""
echo "========================================"
echo "=== Production Verification Summary ==="
echo "========================================"
echo -e "Passed:   ${GREEN}$PASSED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}=== All Required Checks Passed ===${NC}"
    exit 0
else
    echo -e "${RED}=== Some Checks Failed ===${NC}"
    echo "Please fix the failed checks before deploying to production."
    exit 1
fi
